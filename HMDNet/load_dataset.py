import cv2
import numpy as np
from keras import backend as K
from keras.utils import np_utils
import sys, os

path = '/home/graydove/Datasets/PCCD2/'
num_train_samples = 3635
num_val_samples = 300

def load_iamge(file_name, label_key='labels'):
	image = cv2.imread(file_name)
	data = cv2.resize(image, (224,224))
	return data

def load_data():
	x_train = np.empty((num_train_samples, 224, 224, 3), dtype='uint8')
	y_train = np.empty((num_train_samples, 7), dtype='float32')
	x_test = np.empty((num_val_samples, 224, 224, 3), dtype='uint8')
	y_test = np.empty((num_val_samples, 7), dtype='float32')

	fpath = os.path.join(path, 'train')
	train_label = open(os.path.join(path, 'train.txt'), 'r')
	cnt = 0
	for line in train_label.readlines():
		items = line.strip().split(' ')
		x_train[cnt, :, :, :] = load_iamge(os.path.join(path, 'train', items[0]))
		for i in range(len(items) - 1):
			y_train[cnt, i] = float(items[i + 1])
		cnt += 1
	train_label.close()
	fpath = os.path.join(path, 'val')
	val_label = open(os.path.join(path, 'val.txt'), 'r')
	cnt = 0
	for line in val_label.readlines():
		items = line.strip().split(' ')
		x_test[cnt, :, :, :] = load_iamge(os.path.join(path, 'val', items[0]))
		for i in range(len(items) - 1):
			y_test[cnt, i] = float(items[i + 1])
		cnt += 1
	val_label.close()

	return (x_train, y_train), (x_test, y_test)

def load_our_data(img_rows, img_cols):

	# Load our training and validation sets
	(X_train, Y_train), (X_valid, Y_valid) = load_data()

	return X_train, Y_train, X_valid, Y_valid
