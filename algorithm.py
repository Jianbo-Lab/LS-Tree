from itertools import chain, combinations
# from keras.utils.np_utils import to_categorical 
from keras.utils import to_categorical 
from keras.preprocessing import sequence
import numpy as np
import itertools
import scipy.special
from time import time 
import pickle as pkl
from nltk.tokenize import sent_tokenize

from scipy import stats
import pickle as pkl
from keras.preprocessing.text import Tokenizer

import re
import os
from itertools import chain
import nltk
from nltk import Tree
from nltk.parse import CoreNLPParser
from load_data import convert_raw
def explain_instance(raw_text, model, tokenizer, parser = None, phrase = False, text_id = None, data_type = None):
	dataset_name, model_name = model.dataset_name, model.model_name
	# clean text. 
	raw_text = clean_text(raw_text)

	# get parse tree.
	adj_lists, all_nodes, full, if_error = get_parse_tree(dataset_name, text_id, data_type, raw_text, parser)
	nodes = list(chain.from_iterable(all_nodes))
	num_features = len(full)

	# evaluate model
	vals, category = evaluate_model(model, full, nodes, model_name, tokenizer)

	# find init_beta and vals. 
	init_A_inv, init_beta = find_init_A_beta(vals, nodes, num_features)

	# compute interaction scores.
	node_to_score = detect_interaction(vals, adj_lists, init_A_inv, init_beta, num_features)
	
	correct = not if_error
	init_beta = init_beta.reshape((-1))
	return init_beta, node_to_score, nodes, full, adj_lists, category, correct




####################### Functions to find parse tree #################### 
def find_first_sub_list(sl,l): 
	sll=len(sl)
	for ind in (i for i,e in enumerate(l) if e==sl[0]):
		if l[ind:ind+sll]==sl:
			return (ind,ind+sll)  

def find_subtrees_recursive(tree, sent, sent_start_idx, idx_root, adj_list, cur_idx, nodes): 
	for child in tree:
		if isinstance(child, Tree): 
			word_list = list(child.flatten())
			s, e = find_first_sub_list(word_list, sent[cur_idx[0]:]) 
			idx = (s + cur_idx[0] + sent_start_idx, e + cur_idx[0] + sent_start_idx)

			if idx != idx_root:
				adj_list[idx] = []
				adj_list[idx_root].append(idx)
				nodes.add(idx) 

			find_subtrees_recursive(child, sent, sent_start_idx, idx, adj_list, cur_idx, nodes)
			if idx[1] - idx[0] == 1:
				cur_idx[0] = idx[1] - sent_start_idx

def find_subtrees(tree, sent, sent_idx): 
	word_list = list(tree.flatten())  
	s, e = find_first_sub_list(word_list, sent) 
	idx_root = (s + sent_idx[0], e + sent_idx[0])
	adj_list = {}
	adj_list[idx_root] = []
	nodes = {idx_root} 
	find_subtrees_recursive(tree, sent, sent_idx[0], idx_root, adj_list, [0], nodes) 
	return adj_list, nodes
					
def extract_subtrees_of_parse(raw_text, parser = None):
	"""
	Input: string
	Output:
	A list of lists of lists of words. The words in a single inner list form a phrase. 
	"""
	# print(raw_text)
	# print('Is it a unicode?')
	# hahahahhaha
	sents_text = sent_tokenize(raw_text)
	if_error = False
	
	if parser is None:
		parser = CoreNLPParser(url='http://localhost:9000')

	parse_trees = []
	full = [] 
	sent_idx = [] 
	sents_lst = []
	for i, sent in enumerate(sents_text):
		try:  
			const_parse = next(parser.raw_parse(sent))
			sent = list(list(const_parse.subtrees())[0].flatten()) 
			
		except:
			print('There is an error...') 
			print(sent.encode('ascii'))
			sent = nltk.word_tokenize(sent)
			const_parse = sent[:]
			if_error = True


		parse_trees.append(const_parse)
		
		full += sent
		sents_lst.append(sent)
		if i == 0:
			sent_idx.append([0, len(sent)])
		else:
			sent_idx.append([sent_idx[-1][-1], sent_idx[-1][-1] + len(sent)])

	adj_lists = []; all_nodes = []
	for i, const_parse in enumerate(parse_trees): 
		if type(const_parse) == list:

			sent = sents_lst[i]
			start_id, end_id = sent_idx[i] 
			d = len(sent)
			if d == 1:
				adj_list = {(start_id, start_id+1): []}
				nodes = [(start_id, start_id+1)]
			else:
				nodes=[(i+start_id, i+start_id+1) for i in range(d)]
				adj_list = {(start_id, end_id): nodes[:]}	
				for node in nodes:
					adj_list[node] = []
				nodes.append((start_id, end_id))		
		else:
			adj_list, nodes = find_subtrees(const_parse, sents_lst[i],sent_idx[i]) 
			#here print([full[node[0]:node[1]] for node in nodes])

		adj_lists.append(adj_list)
		all_nodes.append(nodes)


	return adj_lists, all_nodes, full, if_error


def get_parse_tree(dataset_name, text_id, data_type, raw_text, parser):
	if dataset_name and text_id is not None and data_type:
		dir_name = os.path.join('./parse_trees', dataset_name)

		if 'parse_trees' not in os.listdir('./'):
			os.mkdir('parse_trees')

		if dataset_name not in os.listdir('./parse_trees'):
			os.mkdir(dir_name)

		filename = 'tree-{}-{}.pkl'.format(data_type, text_id)
		if os.path.isfile(os.path.join(dir_name, filename)):
			with open(os.path.join(dir_name, filename), 'rb') as f:
				adj_lists,all_nodes,full,if_error = pkl.load(f)
		else:
			adj_lists,all_nodes,full,if_error = extract_subtrees_of_parse(raw_text, parser)
			with open(os.path.join(dir_name, filename), 'wb') as f:
				pkl.dump([adj_lists,all_nodes,full,if_error], f)

	else:
		adj_lists, all_nodes, full, if_error = extract_subtrees_of_parse(raw_text, parser)

	return adj_lists, all_nodes, full, if_error

######################################################################



def clean_text(raw_text):
	raw_text = re.sub('\.', '. ', raw_text)
	raw_text = re.sub('\\n', ' ', raw_text)
	raw_text = re.sub('\\\\n', ' ', raw_text)
	raw_text = re.sub('\t', ' ', raw_text)
	# raw_text = re.sub('\\', ' ', raw_text)
	raw_text = re.sub('\'',' ', raw_text)
	raw_text = re.sub('\`',' ', raw_text)
	raw_text = re.sub('\\\\',' ', raw_text)
	return raw_text

def detect_interaction(vals, adj_lists, init_A_inv, init_beta, num_features):

	node_to_score = {}

	for adj_list in adj_lists:
		# find root.
		root = max(adj_list.keys(), key = lambda node: node[1] - node[0])

		# assign interaction recursively.
		detect_interaction_recursive(vals, adj_list, root, init_A_inv, init_beta, num_features, node_to_score)


	return node_to_score


def detect_interaction_recursive(vals, adj_list, node, prev_A_inv, prev_beta, num_features, node_to_score):
	"""
	Inputs:
	vals: dictionary from node to model evaluation. 
	node: the current node.
	prev_A_inv: the inverse matrix of matrix including the node, 
	but not its ancestors. 
	prev_beta: the linear regression estimates including the node, 
	but not its ancestors.
	num_features: number of features.
	node_to_score: dictionary to udpate. 
	"""
	if node[1] - node[0] > 1:
		pos = np.zeros((num_features, 1))
		pos[node[0]:node[1]] = 1
		y = vals[node]

		prev_A_inv_dot_pos = prev_A_inv.dot(pos)

		cur_A_inv = prev_A_inv + prev_A_inv_dot_pos.dot(prev_A_inv_dot_pos.T) / (1 - pos.T.dot(prev_A_inv_dot_pos))

		cur_beta = prev_beta - cur_A_inv.dot(pos) * (y - np.sum(pos * prev_beta))
		# print(cur_A_inv)
		# print(pos.reshape(-1))
		# print(prev_beta.reshape(-1))
		# print(cur_beta.reshape(-1))

		beta_diff = prev_beta - cur_beta
		abs_score, signed_score = np.linalg.norm(beta_diff), np.sum(beta_diff)
		node_to_score[node] = (abs_score, signed_score)

		for child in adj_list[node]:
			detect_interaction_recursive(vals, adj_list, child, cur_A_inv, cur_beta, num_features, node_to_score)
	else:
		beta_node = prev_beta[node[0], 0]
		node_to_score[node] = (abs(beta_node), beta_node)


def evaluate_model(model, full, nodes, model_name, tokenizer):
	inputs = [full[node[0]:node[1]] for node in nodes]
	inputs.append(full)
	inputs.append([])

	inputs = convert_raw(inputs, model_name, tokenizer) 
	probs = model.predict(inputs)
	vals = probs[:-2] - probs[-1]
	category = np.argmax(probs[-2]) 
	vals = vals[:, category]
	vals = {nodes[i]: val for i, val in enumerate(vals)}
	return vals, category

def find_init_A_beta(vals, nodes, num_features):
	num_evals = len(nodes)
	positions = np.zeros((num_evals, num_features))
	y = np.zeros((num_evals, 1))
	for i, node in enumerate(nodes):
		positions[i, nodes[i][0]:nodes[i][1]] = 1
		y[i, 0] = vals[node]

	A = np.transpose(positions).dot(positions)
	b = np.transpose(positions).dot(y)   

	A_inv = np.linalg.inv(A)
	beta = A_inv.dot(b)
	return A_inv, beta





