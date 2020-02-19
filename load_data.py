from keras.datasets import imdb
import numpy as np 
from keras.preprocessing import sequence 
import pickle 
import os
import sys
import tensorflow as tf 
import csv
sys.path.append('./bert') 
import tokenization
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer 
import pandas as pd

class Data():
	def __init__(self, dataset_name, train = False): 
		assert dataset_name == 'sst'

		def read_tsv(input_file, quotechar=None):
			with tf.gfile.Open(input_file, "r") as f:
				reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
				texts, labels = [], []
				for i, line in enumerate(reader):
					if i > 0:
						text = tokenization.convert_to_unicode(line[0])
						label = tokenization.convert_to_unicode(line[1])

						texts.append(text)
						labels.append(int(label))

			labels = to_categorical(np.asarray(labels))
			
			return texts, labels

		texts_train, labels_train = read_tsv('./bert/glue_data/SST-2/train.tsv')
		texts_dev, labels_dev = read_tsv('./bert/glue_data/SST-2/dev.tsv')

		self.x_train_raw = texts_train
		self.y_train = labels_train
		self.x_val_raw = texts_dev
		self.y_val = labels_dev
		self.tokenize(texts_train, texts_dev)

	def tokenize(self, texts_train, texts_dev): 
		print('Fitting tokenizer...')
 
		tokenizer = Tokenizer()
		lt = [t.lower() for t in texts_train]
		tokenizer.fit_on_texts(lt)
		num_words = max(tokenizer.word_index.values())
		print('Number of words: {}'.format(num_words))
		if 'sstcnn' not in os.listdir('./'):
			os.mkdir('sstcnn')
		with open('sstcnn/tokenizer.pkl', 'wb') as f:
			pickle.dump(tokenizer, f)   

		self.tokenizer = tokenizer
		def lt_to_int(lt):
			data = tokenizer.texts_to_sequences(lt)
			avg_len = np.mean([len(l) for l in data])
			max_len = np.max([len(l) for l in data])
			print('The average length is {}'.format(avg_len))
			print('The max length is {}'.format(max_len))
			data = np.array(data)           
			data = sequence.pad_sequences(data, maxlen=50)
			return data

		id_to_word = {tokenizer.word_index[word]: word for word in tokenizer.word_index}

		self.x_train = lt_to_int([t.lower() for t in texts_train])
		self.x_val = lt_to_int([t.lower() for t in texts_dev]) 
		self.lt_to_int = lt_to_int


def load_sst_train():
	# load only sentences of the training data. 
	data_dir = './bert/glue_data/SST-2/original/'

	df_sent = pd.read_csv(os.path.join(data_dir, 'datasetSentences.txt'), delimiter="\t")
	df_label = pd.read_csv(os.path.join(data_dir, 'sentiment_labels.txt'), delimiter="|")
	df_splitlabel = pd.read_csv(os.path.join(data_dir, 'datasetSplit.txt'), delimiter=",")
	df_dictionary = pd.read_csv(os.path.join(data_dir, 'dictionary.txt'), delimiter="|", names =['sentence', 'phrase ids'])

	df = pd.merge(df_sent, df_dictionary, on='sentence', how='left')
	df = pd.merge(df, df_splitlabel, on='sentence_index')
	df = pd.merge(df, df_label, on='phrase ids', how='left')

	def classify(sent_value):
		classification = 0
		if sent_value <= .4:
			classification = 0
		elif sent_value > .6:
			classification = 1
		else:
			classification = 3
			
		return classification

	df['label'] = df['sentiment values'].apply(classify)
	df = df[df['label'] != 3]
	df = df[df['splitset_label'] == 1]
	df = df[['sentence', 'label']]
	df['sentence'] = df['sentence'].apply(tokenization.convert_to_unicode)
	text_label_pairs = df.values.tolist()
	texts, labels = [], []
	for pair in text_label_pairs:
		sent, label = pair
		texts.append(sent)
		labels.append(int(label))

	labels = to_categorical(np.asarray(labels))

	return texts, labels


def open_tokenizer(dataset_name): 
	if dataset_name == 'imdb':
		with open('imdbcnn/data/tokenizer.pkl','rb') as f:
			tokenizer = pickle.load(f) 
			tokenizer.oov_token = None

	elif dataset_name == 'sst':
		with open('sstcnn/tokenizer.pkl','rb') as f:
			tokenizer = pickle.load(f) 	

	elif dataset_name == 'yelp':
		with open('yelpcnn/tokenizer.pkl','rb') as f:
			tokenizer = pickle.load(f) 		
	else:
		tokenizer = None

	return tokenizer


def convert_raw(inputs, model_name, tokenizer):
	# convert raw text to appropriate format for the model.

	if model_name in ['bow', 'bert']:
		outputs_int = [' '.join(i) for i in inputs]

	elif model_name in ['cnn', 'lstm']:
		inputs = [' '.join(i) for i in inputs]
		inputs = [t.lower() for t in inputs]
		def lt_to_int(lt):
			data = tokenizer.texts_to_sequences(lt)
			avg_len = np.mean([len(l) for l in data])
			max_len = np.max([len(l) for l in data])
			
			data = np.array(data)			
			data = sequence.pad_sequences(data, maxlen=50)
			return data

		outputs_int = lt_to_int(inputs)

	return outputs_int
