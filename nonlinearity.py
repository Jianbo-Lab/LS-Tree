import numpy as np 
from load_data import Data, open_tokenizer
from build_model import TextModel
import time
import os
import math
import pickle as pkl
import tensorflow as tf 
from keras import backend as K
from nltk.parse import CoreNLPParser
import matplotlib
import matplotlib.pyplot as plt 
from pylab import boxplot, setp
from scipy.stats.stats import pearsonr

def find_roots(adj_lists):
	roots = []
	all_nodes = [adj_list.keys() for adj_list in adj_lists]
	for i, nodes in enumerate(all_nodes):
		root = sorted(nodes, key = lambda node: node[1] - node[0], 
			reverse = True)[0]
		roots.append(root)

	return roots

def find_depth(adj_lists):
	depths = {}

	roots = find_roots(adj_lists)
	for i, root in enumerate(roots): 
		find_depth_tree(root, adj_lists[i], depths)

	return depths

def find_depth_tree(node, adj_list, depths):

	if node not in depths:
		if node not in adj_list or len(adj_list[node]) == 0 or adj_list[node][0] == node:
			depths[node] = 1
		else:
			children = adj_list[node]
			# print(children)
			for child in children:
				find_depth_tree(child, adj_list, depths)
			depths[node] = 1 + max([depths[child] for child in children])

def compute_average_depth(dataset_name, model_name):
	########### Compute average depth ##########
	data_model = dataset_name + model_name   
	avg_depth = {i:[] for i in range(1, 11)}

	info_filename ='{}/results/info-0-800.pkl'.format(data_model)

	with open(info_filename, 'rb') as f:
		all_info = pkl.load(f) 
		
	out = {}

	for i in range(len(all_info)):
		phis_word, node_to_d, nodes, full, adj_lists, predicted, correct, truth, idx = all_info[i]
		depths = find_depth(adj_lists)

		sorted_nodes = sorted(nodes, key = lambda node: -node_to_d[node][0])

		outputs = []
		
		for k in range(1, 11): 
			avg_depth[k].append(np.mean([depths[node] for node in sorted_nodes[:k]]))

	return avg_depth 

def compute_linear_cor(dataset_name):
	########### Compute average correlation with BoW ##########
	model_names = ['bow', 'cnn', 'lstm', 'bert']
	model_to_label = {'bow': 'BoW','cnn': 'CNN', 'lstm': 'LSTM', 'bert': 'BERT'}


	for model_name in model_names:
		data_model = dataset_name + model_name
		info_filename ='{}/results/info-0-800.pkl'.format(data_model)

		with open(info_filename, 'rb') as f:
			all_info = pkl.load(f) 
			

		with open('{}/vectorizer.pkl'.format('{}bow'.format(dataset_name)), 'rb') as f:
			vectorizer = pkl.load(f)

		with open('{}/clf.pkl'.format('{}bow'.format(dataset_name)), 'rb') as f:
			clf = pkl.load(f)	

		word_to_id = vectorizer.vocabulary_
		coefficients = clf.coef_[0]
		linear_coeffs_dict = {word: coefficients[word_to_id[word]] for word in word_to_id}

		cors = []
		out = {}
		
		for i, info in enumerate(all_info):
			scores_dict = dict()
			phis_word, node_to_d, nodes, full, adj_lists, predicted, correct, truth, idx = info
			for j, word in enumerate(full):
				score = phis_word[j]
				score = -score if predicted == 0 else score

				if word in scores_dict:
					scores_dict[word].append(score)
				else:
					scores_dict[word] = [score]
			scores = []
			linear_coeffs = []				
			for word in scores_dict:
				if word in linear_coeffs_dict:
					scores.append(np.mean(scores_dict[word]))
					linear_coeffs.append(linear_coeffs_dict[word])
			if len(scores) > 2:
				cor = pearsonr(scores, linear_coeffs)[0]
				cors.append(cor)
				out[i] = cor

		print('{} has an avg. correlation of {} with BoW'.format(model_to_label[model_name], np.around(np.mean(cors), 3)))

def plot_depth(dataset_name):
	# plot.
	font = {'weight' : 'bold',
	'size'   : 15}
	matplotlib.rc('font', **font)
	model_names = ['bow', 'cnn', 'lstm', 'bert']
	linestyles = [':','-.','--','-']
	model_to_label = {'bow': 'BoW','cnn': 'CNN', 'lstm': 'LSTM', 'bert': 'BERT'}

	average_depth = {}
	for j, model_name in enumerate(model_names):
		print('Computing average depth for {}'.format(model_name))
		avg_depth = compute_average_depth(dataset_name, model_name)
		average_depth[model_name] = avg_depth
		# for k in range(1, 11): 
		# 	print('{}: {}'.format(k, np.mean(avg_depth[k])))

	plt.figure(figsize = (10, 8))
	plt.tight_layout()
	plt.xlabel('# Top Nodes', fontsize = 32)
	plt.ylabel('Depth', fontsize = 32)
	plt.title('Avg. Depth vs. # Top Nodes (SST)', fontsize = 32)
	plt.ylim(1, 3)
	plt.xticks(np.arange(1, 11), fontsize = 15)
	plt.yticks(np.arange(1, 3 + 1, 0.5), fontsize = 15)

	for j, model_name in enumerate(model_names): 
		plt.plot(range(1, 11), [np.mean(average_depth[model_name][k]) for k in range(1, 11)], label = model_to_label[model_name], linewidth = 4, linestyle = linestyles[j]) 
	if not os.path.exists('figs'):
		os.mkdir('figs')

	plt.savefig('figs/avg_depth-{}.pdf'.format(dataset_name))
	plt.close()


if __name__ == '__main__':
	compute_linear_cor('sst')
	plot_depth('sst')








