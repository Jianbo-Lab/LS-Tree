import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Permute,MaxPooling1D, Flatten, LSTM, Bidirectional, GRU, GlobalAveragePooling1D, SeparableConv1D, CuDNNLSTM

from keras.datasets import imdb
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy as accuracy
from keras.optimizers import RMSprop, Adam
from keras import backend as K  
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
import os, itertools, math
import pickle
import sys
def construct_original_network(emb, model_name, trainable=True): 
	if model_name == 'cnn':
		print("Creating Model...")
		from keras.layers.merge import concatenate

		num_filters = 100
		kernel_sizes = [3,4,5]
		conv_0 = Conv1D(num_filters,
						 kernel_sizes[0],
						 padding='valid',
						 activation='relu',
						 kernel_initializer='normal', 
						 strides=1, 
						 trainable=trainable)(emb)
		conv_1 = Conv1D(num_filters,
						 kernel_sizes[1],
						 padding='valid',
						 activation='relu',
						 kernel_initializer='normal', 
						 strides=1, 
						 trainable=trainable)(emb)
		conv_2 = Conv1D(num_filters,
						 kernel_sizes[2],
						 padding='valid',
						 activation='relu',
						 kernel_initializer='normal', 
						 strides=1, 
						 trainable=trainable)(emb) 

		maxpool_0 = GlobalMaxPooling1D()(conv_0)
		maxpool_1 = GlobalMaxPooling1D()(conv_1)
		maxpool_2 = GlobalMaxPooling1D()(conv_2)

		concatenated_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2])

		hidden_dims = 200 
		concatenated_tensor = Dense(hidden_dims, trainable = trainable, activation = 'relu')(concatenated_tensor)

		dropout = Dropout(0.5)(concatenated_tensor)
		preds = Dense(units=2, activation='softmax')(dropout)

		return preds


	elif model_name == 'lstm': 
		net = Dropout(0.5, name = 'dropout_1')(emb)		
		net = Bidirectional(CuDNNLSTM(128,trainable=trainable), trainable = trainable)(net)  

		net = Dropout(0.5, name = 'dropout_2')(net) 

		net = Dense(2, name = 'dense_2',trainable=trainable)(net)
		preds = Activation('softmax')(net)
		return preds  

class TextModel():
	def __init__(self, model_name, dataset_name, train = False, **kwargs):
		self.model_name = model_name
		self.dataset_name = dataset_name
		self.data_model = dataset_name + model_name

		if model_name in ['cnn', 'lstm']:
			self.maxlen = 50
			self.num_words = 13822 + 1 

			self.embedding_dims = 300
			self.num_classes = 2

			if not train:
				K.set_learning_phase(0)

			X_ph = Input(shape=(self.maxlen,), dtype='int32')

			if train:
				emb_file = '{}cnn/embedding_matrix.npy'.format(dataset_name)

				embedding_matrix = np.load(emb_file)
				emb_layer = Embedding(self.num_words, self.embedding_dims,
					input_length=self.maxlen, 
					weights=[embedding_matrix], 
					name = 'embedding_1', 
					trainable = True)
				
				emb_out = emb_layer(X_ph)
				preds = construct_original_network(emb_out, model_name)	

				pred_model = Model(X_ph, preds)
				
				pred_model.compile(loss='categorical_crossentropy',
							  optimizer='adam',
							  metrics=['accuracy']) 
				self.pred_model = pred_model 

			else: 
				emb_layer = Embedding(self.num_words, self.embedding_dims,
					input_length=self.maxlen, name = 'embedding_1', trainable = True)
				emb_out = emb_layer(X_ph)
				emb_ph = Input(shape=(self.maxlen, self.embedding_dims), 
					dtype='float32')   
				preds = construct_original_network(emb_ph, model_name) 

				model1 = Model(X_ph, emb_out)
				model2 = Model(emb_ph, preds) 
				pred_out = model2(model1(X_ph))   


				pred_model = Model(X_ph, pred_out) 
				pred_model.compile(loss='categorical_crossentropy',
							  optimizer='adam',
							  metrics=['accuracy']) 
				self.pred_model = pred_model 

				self.sess = K.get_session()  

				self.input_ph = X_ph
				self.emb_out = emb_out
				self.emb_ph = emb_ph
				weights_name = 'original.h5'
				model1.load_weights('{}/models/{}'.format(self.data_model, weights_name), 
					by_name=True)
				model2.load_weights('{}/models/{}'.format(self.data_model, weights_name), 
					by_name=True)  
				self.model1 = model1
				self.model2 = model2
				print('Model constructed.') 


		elif model_name == 'bert':
			
			sys.path.append('./bert')

			from bert_model import construct_bert_predictor, predict

			self.maxlen = {'imdbbert': 350, 'sstbert': 128, 'yelpbert': 256}[self.data_model]
			if 'checkpoint' not in kwargs:
				checkpoint = None
			else:
				checkpoint = kwargs['checkpoint']

			self.processor, self.estimator, self.tokenizer = construct_bert_predictor(init_checkpoint = checkpoint, 
				dataset_name = dataset_name)

			self.predict_fn = lambda xs: predict(xs, self.maxlen, self.processor, self.estimator, self.tokenizer, dataset_name)

		elif model_name == 'bow':
			with open('{}/vectorizer.pkl'.format(self.data_model), 'rb') as f:
				self.vectorizer = pickle.load(f)


			with open('{}/clf.pkl'.format(self.data_model), 'rb') as f:
				self.clf = pickle.load(f)			

	def train(self, dataset): 
		assert self.model_name in ['cnn', 'lstm']
		if self.dataset_name == 'sst':
			epochs = 10
			batch_size = 30 
			filepath = '{}/models/original.h5'.format(self.data_model)
			checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
				verbose=1, save_best_only=True, mode='max')

			# callbacks_list = [checkpoint] 

			def step_decay(epoch):
				initial_lrate = 1e-3
				drop = 0.9
				epochs_drop = 1.0
				lrate = initial_lrate * math.pow(drop,  
				math.floor((1+epoch)/epochs_drop))
				return lrate

			lrate = LearningRateScheduler(step_decay)
			callbacks_list = [checkpoint, lrate] 

			self.pred_model.fit(dataset.x_train, dataset.y_train, validation_data=(dataset.x_val, dataset.y_val),callbacks = callbacks_list, epochs=epochs, batch_size=batch_size)


	def predict(self, x, verbose=0):
		if self.model_name in ['cnn', 'lstm']: 
			if type(x) == list or x.shape[1] < self.maxlen:
				x = np.array(sequence.pad_sequences(x, maxlen=self.maxlen)) 
			batch_size = 100
			prob = self.pred_model.predict(x, batch_size = batch_size, 
				verbose = verbose) 
			return np.log(prob) 

		elif self.model_name == 'bert': 
			prob = self.predict_fn(x)

			return np.log(prob) 

		elif self.model_name == 'bow':
			x = self.vectorizer.transform(x)
			pred = self.clf.predict_proba(x)+np.finfo(float).resolution
			dec = np.log(pred[:, 0] / (1 - pred[:, 0]))
			pred = np.array([dec, -dec]).T
			return pred 