from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import csv
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical 
import sys
from load_data import Data 
from sklearn.linear_model import LogisticRegression
import os
import pickle as pkl

def train_bow(dataset_name):
	# Train the BoW model. 
	data_model = dataset_name + 'bow'
	
	dataset = Data(dataset_name)
	x_train, y_train = dataset.x_train_raw, np.argmax(dataset.y_train, axis = 1)
	x_test, y_test = dataset.x_val_raw, np.argmax(dataset.y_val, axis = 1)

	print('Fitting transform...')
	vectorizer = CountVectorizer(max_features = 20000)
	x_train_bow = vectorizer.fit_transform(x_train)

	x_test_bow = vectorizer.transform(x_test)

	
	print('Fitting logistic regression...')
	clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial') 
	clf.fit(x_train_bow, y_train)

	print('Making prediction...')
	pred_test = clf.predict_proba(x_test_bow)
	acc_train = clf.score(x_train_bow, y_train)
	acc_test = clf.score(x_test_bow, y_test)
	print('The training accuracy is {}; the test accuracy is {}.'.format(acc_train, acc_test))

	# print('The size of bow transformer is {} MB.'.format(sys.getsizeof(vectorizer) * 1e-6))

	print('Save model to pickle...')
	
	if data_model not in os.listdir('.'):
		os.mkdir(data_model)

	with open('{}/vectorizer.pkl'.format(data_model), 'wb') as f:
		pkl.dump(vectorizer, f)


	with open('{}/clf.pkl'.format(data_model), 'wb') as f:
		pkl.dump(clf, f)	


if __name__ == '__main__':
	train_bow('sst')












