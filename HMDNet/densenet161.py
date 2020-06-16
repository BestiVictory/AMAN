# -*- coding: utf-8 -*-

from keras.optimizers import SGD
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.models import Model
import keras.backend as K
from keras import losses
from keras.models import load_model

from sklearn.metrics import log_loss

from custom_layers.scale_layer import Scale

from load_dataset import load_our_data
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

def densenet161_model(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.5, dropout_rate=0, weight_decay=1e-4, weights_file=None):

	eps = 1.1e-5

	# compute compression factor
	compression = 1.0 - reduction

	# Handle Dimension Ordering for different backends
	global concat_axis
	if K.image_dim_ordering() == 'tf':
	  concat_axis = 3
	  img_input = Input(shape=(224, 224, 3), name='data')
	else:
	  concat_axis = 1
	  img_input = Input(shape=(3, 224, 224), name='data')

	# From architecture for ImageNet (Table 1 in the paper)
	nb_filter = 96
	nb_layers = [6,12,36,24] # For DenseNet-161

	# Initial convolution
	x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
	x = Conv2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
	x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
	x = Scale(axis=concat_axis, name='conv1_scale')(x)
	x = Activation('relu', name='relu1')(x)
	x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
	x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

	# Add dense blocks
	for block_idx in range(nb_dense_block - 1):
		stage = block_idx+2
		x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

		# Add transition_block
		x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
		nb_filter = int(nb_filter * compression)

	final_stage = stage + 1
	x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

	x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
	x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
	x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
	
	model = Model(img_input, x, name='densenet')
	# model = keras.applications.mobilenet.MobileNet()

	if K.image_dim_ordering() == 'th':
	  # Use pre-trained weights for Theano backend
	  weights_path = 'imagenet_models/densenet161_weights_th.h5'
	else:
	  # Use pre-trained weights for Tensorflow backend
	  weights_path = 'imagenet_models/densenet161_weights_tf.h5'

	if weights_file is None:
		model.load_weights(weights_path, by_name=True)

	# Truncate and replace softmax layer for transfer learning
	# Cannot use model.layers.pop() since model is not of Sequential() type
	# The method below works since pre-trained weights are stored in layers but not in the model
	x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
	
	# Our Hierarchical Mult-task
	# Part 1. Global regression
	global_fc = Dense(512, name='global_feature')(x_newfc)
	global_fc = Dense(1, name='global_fc')(global_fc)
	global_fc = Activation('sigmoid', name='global_loss')(global_fc)
	
	# Part 2. Attribute regression
	attribute_fc = Dense(512, name='attribute_feature')(x_newfc)
	
	attribute1_fc = Dense(256, name='attribute1_feature')(attribute_fc)
	attribute1_fc = Dense(1, name='attribute1_fc')(attribute1_fc)
	attribute1_fc = Activation('sigmoid', name='attribute1_loss')(attribute1_fc)
	
	attribute2_fc = Dense(256, name='attribute2_feature')(attribute_fc)
	attribute2_fc = Dense(1, name='attribute2_fc')(attribute2_fc)
	attribute2_fc = Activation('sigmoid', name='attribute2_loss')(attribute2_fc)
	
	attribute3_fc = Dense(256, name='attribute3_feature')(attribute_fc)
	attribute3_fc = Dense(1, name='attribute3_fc')(attribute3_fc)
	attribute3_fc = Activation('sigmoid', name='attribute3_loss')(attribute3_fc)
	
	attribute4_fc = Dense(256, name='attribute4_feature')(attribute_fc)
	attribute4_fc = Dense(1, name='attribute4_fc')(attribute4_fc)
	attribute4_fc = Activation('sigmoid', name='attribute4_loss')(attribute4_fc)
	
	attribute5_fc = Dense(256, name='attribute5_feature')(attribute_fc)
	attribute5_fc = Dense(1, name='attribute5_fc')(attribute5_fc)
	attribute5_fc = Activation('sigmoid', name='attribute5_loss')(attribute5_fc)
	
	attribute6_fc = Dense(256, name='attribute6_feature')(attribute_fc)
	attribute6_fc = Dense(1, name='attribute6_fc')(attribute6_fc)
	attribute6_fc = Activation('sigmoid', name='attribute6_loss')(attribute6_fc)

	model = Model(img_input, [global_fc, attribute1_fc, attribute2_fc, attribute3_fc, attribute4_fc, attribute5_fc, attribute6_fc])
	
	if weights_file is not None:
		model.load_weights(weights_file)
	# Learning rate is changed to 0.001
	sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss=losses.mean_squared_error)

	return model
	
def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
	eps = 1.1e-5
	conv_name_base = 'conv' + str(stage) + '_' + str(branch)
	relu_name_base = 'relu' + str(stage) + '_' + str(branch)

	# 1x1 Convolution (Bottleneck layer)
	inter_channel = nb_filter * 4  
	x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
	x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
	x = Activation('relu', name=relu_name_base+'_x1')(x)
	# x = Conv2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)
	x = Conv2D(inter_channel, (1, 1), use_bias=False, name=conv_name_base+'_x1')(x)

	if dropout_rate:
		x = Dropout(dropout_rate)(x)

	# 3x3 Convolution
	x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
	x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
	x = Activation('relu', name=relu_name_base+'_x2')(x)
	x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
	# x = Conv2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False)(x)
	x = Conv2D(nb_filter, (3, 3), use_bias=False, name=conv_name_base+'_x2')(x)
	
	if dropout_rate:
		x = Dropout(dropout_rate)(x)

	return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):

	eps = 1.1e-5
	conv_name_base = 'conv' + str(stage) + '_blk'
	relu_name_base = 'relu' + str(stage) + '_blk'
	pool_name_base = 'pool' + str(stage) 

	x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
	x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
	x = Activation('relu', name=relu_name_base)(x)
	# x = Conv2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)
	x = Conv2D(int(nb_filter * compression), (1, 1), use_bias=False, name=conv_name_base)(x)

	if dropout_rate:
		x = Dropout(dropout_rate)(x)

	x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

	return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
	eps = 1.1e-5
	concat_feat = x

	for i in range(nb_layers):
		branch = i+1
		x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
		concat_feat = concatenate([concat_feat, x], axis = -1, name='concat_'+str(stage)+'_'+str(branch))
		# concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

		if grow_nb_filters:
			nb_filter += growth_rate

	return concat_feat, nb_filter

if __name__ == '__main__':

	img_rows, img_cols = 224, 224 # Resolution of inputs
	channel = 3
	batch_size = 16
	nb_epoch = 25

	# Load our data. Please implement your own load_data() module for your own dataset
	X_train, Y_train, X_valid, Y_valid = load_our_data(img_rows, img_cols)

	# Load our model
	model = densenet161_model(img_rows=img_rows, img_cols=img_cols, color_type=channel)
	
	# Start Fine-tuning
	model.fit(X_train, [Y_train[:,0], Y_train[:,1], Y_train[:,2], Y_train[:,3], Y_train[:,4], Y_train[:,5], Y_train[:,6]],
			batch_size=batch_size,
			nb_epoch=nb_epoch,
			shuffle=True,
			verbose=1,
			validation_data=(X_valid, [Y_valid[:,0], Y_valid[:,1], Y_valid[:,2], Y_valid[:,3], Y_valid[:,4], Y_valid[:,5], Y_valid[:,6]]),
			)
			
	model.save('save_model/our_model_20191223.h5')
	
	# Make predictions
	# print X_valid, type(X_valid), X_valid.shape
	predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

	# MSE loss score
	#score = log_loss(Y_valid, predictions_valid)
