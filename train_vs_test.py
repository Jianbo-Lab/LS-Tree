import numpy as np 
from load_data import Data, open_tokenizer, load_sst_train
from build_model import TextModel
from algorithm import explain_instance
import time
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import math
import pickle as pkl
import tensorflow as tf 
from keras import backend as K
import sys
sys.path.append('./bert') 
import tokenization
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np 
import pandas as pd 
import os
from keras.preprocessing import sequence	
import csv
import matplotlib
import matplotlib.pyplot as plt 
from scipy.stats import kurtosis, skew
from matplotlib.patches import Rectangle
from permutation_test import *

dataset_name = 'sst'
max_model_id = {'cnn': 40, 'bert': 9, 'lstm': 30}
font = {'weight' : 'bold',
	'size'   : 15}
model_names = ['cnn', 'lstm', 'bert']
model_to_label = {'cnn': 'CNN', 'lstm': 'LSTM', 'bert': 'BERT'}
matplotlib.rc('font', **font)
colors = {'train': 'red', 'test': 'blue'}
labels = {'train': 'Train', 'test': 'Test'}
linestyles = {'train': '--', 'test': '-'}
num_data = 500
print('# of data {}'.format(num_data))

def load_info(model_name, data_type, model_id):
	data_model = dataset_name + model_name

	info_filename ='{}/results/info-{}-{}.pkl'.format(data_model, model_id, data_type)

	with open(info_filename, 'rb') as f:
		all_info = pkl.load(f) 

	return all_info

def make_data(dataset_name):
	if dataset_name == 'sst':
		dataset = Data(dataset_name) 
		texts, labels = load_sst_train()

		x_train = dataset.lt_to_int([t.lower() for t in texts])
		y_train = labels
		x_train_raw = texts

		x_test = dataset.x_val
		y_test = dataset.y_val
		x_test_raw = dataset.x_val_raw	

	return x_train, y_train, x_train_raw, x_test, y_test, x_test_raw

def train(args):
	model_name = args.model_name
	data_model = dataset_name + model_name

	assert model_name in ['cnn', 'lstm']

	x_train, y_train, x_train_raw, x_test, y_test, x_test_raw = make_data(dataset_name)
	model = TextModel(model_name, dataset_name, train = True)
	model.pred_model.save_weights(data_model + '/models/train_vs_test-000.h5')
	epochs = max_model_id[model_name]
	batch_size = 100 
	filepath = data_model + '/models/train_vs_test-{epoch:03d}.h5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 
		verbose=1, period=1)

	def step_decay(epoch):
		initial_lrate = 1e-4
		drop = 1.0
		epochs_drop = 10.0
		lrate = initial_lrate * math.pow(drop,  
		math.floor((1+epoch)/epochs_drop))
		return lrate

	lrate = LearningRateScheduler(step_decay)
	callbacks_list = [checkpoint, lrate]

	model.pred_model.fit(x_train, y_train, validation_data=(x_test, y_test),callbacks = callbacks_list, epochs=epochs, batch_size=batch_size)

def generate_scores(args):
	model_name = args.model_name
	data_model = dataset_name + model_name

	from nltk.parse import CoreNLPParser
	parser = CoreNLPParser(url='http://localhost:9000')
	tokenizer = open_tokenizer(dataset_name)
	
	x_train, y_train, x_train_raw, x_test, y_test, x_test_raw = make_data(dataset_name)
	x = {'train': x_train, 'test': x_test}
	x_raw = {'train': x_train_raw, 'test': x_test_raw}
	y = {'train': y_train, 'test': y_test}

	for model_id in range(0, max_model_id[model_name] + 1):
		if model_name in ['cnn', 'lstm']:
			model = TextModel(model_name, dataset_name, train = False)
			model.model1.load_weights(data_model + '/models/train_vs_test-%03d.h5'%model_id, by_name=True)
			model.model2.load_weights(data_model + '/models/train_vs_test-%03d.h5'%model_id, by_name=True)

		elif model_name == 'bert':
			checkpoint = './bert/models/sst_train_vs_test/model.ckpt-{}'.format(int(model_id * 205))

			model = TextModel(model_name, dataset_name, train = False, checkpoint = checkpoint)

		for data_type in ['train','test']:
			print('Generating scores for model {} and {} data'.format(model_id, data_type))
			all_info = []

			for i, sample in enumerate(x_raw[data_type][:num_data]):
				d = len(sample.split(' '))
				if d > 1:	
					truth = np.argmax(y[data_type][i])
					print('explaining the {}th sample...'.format(i))   
					# try: 
					phis_word, node_to_d, nodes, full, adj_lists, predicted, correct = explain_instance(sample, model, tokenizer, parser, True, text_id = i, data_type = data_type)

					all_info.append([phis_word, node_to_d, nodes, full, adj_lists, predicted, correct])

			with open('{}/results/info-{}-{}.pkl'.format(data_model, model_id, data_type), 'wb') as f:
				pkl.dump(all_info, f)

		K.clear_session()

def compute_and_plot(args):
	print('Computing loss...')
	compute_loss(args)
	print('Computing variance...')
	compute_variance(args)
	print('Carrying out permutation test...')
	permutation_test_for_variance(args)
	print('Plotting...')
	plot_loss(args)
	plot_variance(args)
	plot_pvals(args)


def compute_loss(args):
	model_name = args.model_name
	data_model = dataset_name + model_name

	x_train, y_train, x_train_raw, x_test, y_test, x_test_raw = make_data(dataset_name)

	losses = {}
	for model_id in range(0, max_model_id[model_name] + 1): 
		print('Computing loss for model {}'.format(model_id))
		
		if model_name in ['cnn', 'lstm']:
			model = TextModel(model_name, dataset_name, train = False)
			model.model1.load_weights(data_model + '/models/train_vs_test-%03d.h5'%model_id, by_name=True)
			model.model2.load_weights(data_model + '/models/train_vs_test-%03d.h5'%model_id, by_name=True)

			train_loss, _ = model.pred_model.evaluate(x_train, y_train, verbose=0)
			val_loss, _ = model.pred_model.evaluate(x_test, y_test, verbose=0)
			K.clear_session()

		elif model_name == 'bert':
			checkpoint = './bert/models/sst_train_vs_test/model.ckpt-{}'.format(int(model_id * 205))

			model = TextModel(model_name, dataset_name, train = False, checkpoint = checkpoint)
			pred_train = model.predict(x_train_raw)
			pred_val = model.predict(x_test_raw)
			def cross_entropy(predictions, targets, epsilon=1e-12):
				predictions = np.clip(predictions, epsilon, 1. - epsilon)
				N = predictions.shape[0]
				ce = -np.sum(targets*np.log(predictions+1e-9))/N
				return ce


			train_loss, val_loss = cross_entropy(pred_train, y_train), cross_entropy(pred_val, y_test)


			tf.reset_default_graph()

		losses[model_id] = [train_loss, val_loss]

		print('Train loss {:0.2f}; Val loss: {:0.2f}.'.format(train_loss, val_loss))

	with open('{}/results/losses.pkl'.format(data_model), 'wb') as f:
		pkl.dump(losses, f)

def compute_variance(args):
	model_name = args.model_name
	data_model = dataset_name + model_name

	interactions, LS_scores = {}, {}
	data_types = ['train', 'test']
	model_ids = range(0, max_model_id[model_name] + 1)


	variances = {model_id:
					{data_type:[] for data_type in data_types
					} for model_id in model_ids
				}

	for model_id in model_ids: 
		for data_type in data_types:
			all_info = load_info(model_name, data_type, model_id)

			interactions = []
			LS_scores = []

			for i, info in enumerate(all_info):
				phis_word, node_to_d, nodes, full, adj_lists, predicted, correct = info
				score = [val[0] for val in node_to_d.values()]
				variances[model_id][data_type].append(np.var(score))

	with open('{}/results/variances.pkl'.format(data_model), 'wb') as f:
		pkl.dump(variances, f)

def permutation_test_for_variance(args):
	model_name = args.model_name
	data_model = dataset_name + model_name

	model_ids = range(0, max_model_id[model_name] + 1)
	with open('{}/results/variances.pkl'.format(data_model), 'rb') as f:
		variances = pkl.load(f)

	pvals = {}
	for model_id in model_ids:
		var_train = variances[model_id]['train']
		var_test = variances[model_id]['test']
		np.random.seed(0)
		p_value = permutation_test(var_test, var_train, n_iters = 50000)
		print('model_id: {}, pval: {}'.format(model_id, 
			np.round(p_value, 4)))
		print('-----------------------------')
		pvals[model_id] = p_value
		
	with open('{}/results/pvals.pkl'.format(data_model), 'wb') as f:
		pkl.dump(pvals, f)

def plot_pvals(args):
	model_name = args.model_name
	data_model = dataset_name + model_name
	properties = {'weight':'bold'}

	with open('{}/results/pvals.pkl'.format(data_model), 'rb') as f:
		pvals = pkl.load(f)
	model_ids = range(0, max_model_id[model_name] + 1)
	matplotlib.rc('text', usetex=True)
	matplotlib.rc('font',**font)
	matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

	plt.figure(figsize = (10, 8))

	plt.plot(model_ids, 
		[pvals[model_id] for model_id in model_ids], 
		color = 'red',
		label = r'\textbf{P-value}',
		linewidth=4)

	plt.plot(model_ids, 
		[0.05 for model_id in model_ids], 
		linestyle = '--',
		color = 'black',
		label = r'$\alpha = 0.5$',
		linewidth=4)

	plt.legend(fontsize = 32)
	plt.tick_params(axis='both', which='major', labelsize=15)
	plt.tick_params(axis='both', which='minor', labelsize=15)
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel(r'\textbf{Epoch}', fontsize = 32)
	plt.ylabel(r'\textbf{P-value}', fontsize = 32)
	plt.title(r'\textbf{P-value}' + r' \textbf{{vs. Epoch ({})}}'.format(model_to_label[model_name]), fontsize = 32)
	# plt.xlim(left = -1, right = 1)
	plt.tight_layout()
	plt.savefig('figs/pval-{}.pdf'.format(data_model))
	plt.close()

def plot_variance(args):
	model_name = args.model_name
	data_model = dataset_name + model_name
	with open('{}/results/variances.pkl'.format(data_model), 'rb') as f:
		variances = pkl.load(f)

	data_types = ['train', 'test']
	model_ids = range(0, max_model_id[model_name] + 1)

	plt.figure(figsize = (10, 8))

	for data_type in data_types:
		plt.plot(model_ids, 
			[np.mean(variances[model_id][data_type]) for model_id in model_ids], 
			color = colors[data_type], 
			label = labels[data_type],
			linestyle = linestyles[data_type], 
			linewidth=4)

	plt.tick_params(axis='both', which='major', labelsize=15)
	plt.tick_params(axis='both', which='minor', labelsize=15)
	plt.grid(axis='y', alpha=0.75)
	plt.xlabel('Epoch', fontsize = 32)
	plt.ylabel('Variance', fontsize = 32)
	plt.title('Variance' + ' vs. Epoch ({})'.format(model_to_label[model_name]), fontsize = 32)
	plt.tight_layout()
	plt.savefig('figs/variance-{}.pdf'.format(model_name))
	plt.close()	



def plot_loss(args):
	model_name = args.model_name
	data_model = dataset_name + model_name
	data_types = ['train', 'test']
	losses = {data_type: [] for data_type in data_types}
	with open('{}/results/losses.pkl'.format(data_model), 'rb') as f:
		losses = pkl.load(f) 

	model_ids = range(0, max_model_id[model_name] + 1)

	plt.figure(figsize = (10, 8))
	plt.xlabel('Epoch', fontsize = 32)
	plt.ylabel('Loss', fontsize = 32)
	plt.title('Variance vs. Epoch ({})'.format(model_to_label[model_name]), fontsize = 32)

	for data_type_idx, data_type in enumerate(data_types):
		plt.plot(model_ids, [losses[idx][data_type_idx] for idx in model_ids], 
			label = labels[data_type], 
			color = colors[data_type],
			linestyle = linestyles[data_type],
			linewidth = 4)

	plt.tight_layout()
	plt.savefig('figs/loss-{}.pdf'.format(data_model))
	plt.close()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', type = str, 
		choices = ['train', 'generate_scores', 'compute_and_plot'], 
		default = 'compute_and_plot')
	parser.add_argument('--model_name', type = str, 
		choices = ['cnn', 'lstm', 'bert'], 
		default = 'cnn') 	

	args = parser.parse_args()

	tasks = {
				'train': train,
				'generate_scores': generate_scores,
				'compute_and_plot': compute_and_plot
			}

	if not os.path.exists('figs'):
		os.mkdir('figs')

	data_model = dataset_name + args.model_name
	if not os.path.exists(data_model):
		os.mkdir(data_model)
	if not os.path.exists('{}/results'.format(data_model)):
		os.mkdir('{}/results'.format(data_model))
	if not os.path.exists('{}/models'.format(data_model)):
		os.mkdir('{}/models'.format(data_model))

	tasks[args.task](args)



