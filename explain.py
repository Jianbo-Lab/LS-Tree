from __future__ import absolute_import, division, print_function 

import numpy as np
import tensorflow as tf
import os
from keras.utils import to_categorical

import time 
import numpy as np 
import sys
import os
import urllib2 
import tarfile
import zipfile 
import math

from algorithm import explain_instance
from build_model import TextModel 
from load_data import Data, load_sst_train, open_tokenizer
import pickle as pkl
from plot_tree import create_graph

def explain(args):
	dataset, model =  args.dataset, args.model

	st = time.time()
	print('Making explanations...')  

	all_info = []
	from nltk.parse import CoreNLPParser
	parser = CoreNLPParser(url='http://localhost:9000')
	
	tokenizer = open_tokenizer(args.dataset_name)

	samples = dataset.x_val_raw[args.start_id:args.end_id]
	labels =  dataset.y_val

	for i, sample in enumerate(samples): 
		d = len(sample.split(' '))
		if d > 1:
			truth = np.argmax(labels[i+args.start_id])
			print('explaining the {}th sample...'.format(i+args.start_id)) 


			phis_word, node_to_d, nodes, full, adj_lists, predicted, correct =explain_instance(sample, model, tokenizer, parser, True, text_id = i + args.start_id, data_type = 'test')

			classes = ['negative', 'positive']
			print('predicted: {}, truth: {}'.format(classes[predicted], classes[truth]))

			all_info.append([phis_word, node_to_d, nodes, full, adj_lists, predicted, correct, truth, i+args.start_id]) 

	print('Saving to file...')
	with open('{}/results/info-{}-{}.pkl'.format(args.data_model, args.start_id, args.end_id), 'wb') as f:
		pkl.dump(all_info, f)

	return all_info


def demo(args):
	dataset, model =  args.dataset, args.model

	st = time.time()
	print('Making explanations...')  

	all_info = []
	from nltk.parse import CoreNLPParser
	parser = CoreNLPParser(url='http://localhost:9000')
	
	tokenizer = open_tokenizer(args.dataset_name)

	samples = dataset.x_val_raw[args.start_id:args.end_id][:10]
	labels =  dataset.y_val[:10]

	for i, sample in enumerate(samples): 
		d = len(sample.split(' '))
		if d > 1:
			truth = np.argmax(labels[i+args.start_id])
			print('explaining the {}th sample...'.format(i+args.start_id)) 

			phis_word, node_to_d, nodes, full, adj_lists, predicted, correct =explain_instance(sample, model, tokenizer, parser, True, text_id = i + args.start_id, data_type = 'test')

			create_graph(adj_lists, full, node_to_d, 'demo-{}'.format(i), labels = {})






	return all_info
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', type = str, 
		choices = ['train', 'explain', 'demo'], 
		default = 'explain')

	parser.add_argument('--dataset_name', type = str, 
		choices = ['sst'], 
		default = 'sst') 

	parser.add_argument('--model_name', type = str, 
		choices = ['bow', 'cnn', 'lstm', 'bert'], 
		default = 'cnn') 

	parser.add_argument('--start_id', type = int, default = 0)

	parser.add_argument('--end_id', type = int, default = 800)

	args = parser.parse_args()
	dict_a = vars(args)  
	print('Loading dataset...') 
	dataset = Data(args.dataset_name)

	data_model = args.dataset_name + args.model_name


	if data_model not in os.listdir('./'):	
		os.mkdir(data_model)
	if 'results' not in os.listdir('./{}'.format(data_model)):
		os.mkdir('{}/results'.format(data_model))
	if 'models' not in os.listdir(data_model):
		os.mkdir('{}/models'.format(data_model))
			

	if args.task == 'train':
		model = TextModel(args.model_name, args.dataset_name, train = True)

		model.train(dataset) 
		print('Training is done.')
	else:
		model = TextModel(args.model_name, args.dataset_name, train = False)
		dict_a.update({'dataset': dataset, 'model': model, 
			'data_model': data_model})
		if args.task == 'explain':
			explain(args)
		elif args.task == 'demo':
			demo(args)

	











