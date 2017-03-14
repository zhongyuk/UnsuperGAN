import numpy as np
import tensorflow as tf
from utils import *

# latent Z shape
z_dim = 100
G_wt_shapes = {'g_dense1' : [z_dim, 1024],
				'g_dense2': [1024, 7*7*256],
				'g_deconv1': [5, 5, 256, 64],
				'g_deconv2': [5, 5, 64, 1]}


def generator(Z, is_training): # Z latent vector: [batch_size, z_dim]
	# project Z into dense layer1 - Z - [-1, z_dim]
	layer1 = dense(Z, "g_dense1")			#[-1, 1024]
	# Batch Norm
	layer1 = batch_norm(layer1, 'g_dense1', is_training) #[-1, 1024]
	# Activation
	layer1 = activation(layer1, 'leaky') 	#[-1, 1024]

	# Dense layer2
	layer2 = dense(layer1, 'g_dense2')		#[-1, 7*7*256]
	# Batch Norm
	layer2 = batch_norm(layer2, 'g_dense2', is_training) #[-1, 7*7*256]
	# Activation
	layer2 = activation(layer2, 'leaky')  	#[-1, 7*7*256]

	# reshape 
	reshaped = reshape_tensor(layer2, [-1, 7, 7, 256]) #[-1, 7, 7, 256]

	# Deconv 2D
	layer3 = deconv2D(reshaped, 'g_deconv1') 	#[-1, 14, 14, 64]
	# Batch Norm
	layer3 = batch_norm(layer3, 'g_deconv1', is_training) 	#[-1, 14, 14, 64]
	# Activation
	layer3 = activation(layer3, 'leaky')		#[-1, 14, 14, 64]

	# Deconv 2D
	layer4 = deconv2D(layer3, 'g_deconv2')		#[-1, 28, 28, 1]
	# output deconv layer no batch norm??
	layer4 = activation(layer3, 'tanh') # recommend 'tanh' output

	return layer4

num_class = 10
D_wt_shapes = {'d_conv1' : [5, 5, 1, 64],
			   'd_conv2' : [5, 5, 64, 256],
			   'd_dense1' : [7*7*256, 1024],
			   'd_dense2' : [1024, num_class]} # only 2 category....

def discriminator(X, is_training): # X - input image: [batch_size, 28, 28, 1]
	# Conv2D - X: [-1, 28, 28, 1]
	layer1 = conv2D(X, 'd_conv1') 								#[-1, 28, 28, 64]
	layer1 = batch_norm(layer1, 'd_convlayer1', is_training)	#[-1, 28, 28, 64]
	layer1 = activation(layer1, 'leaky')						#[-1, 28, 28, 64]
	layer1 = pooling(layer1, 'avg') 							#[-1, 14, 14, 64]

	# Conv2D
	layer2 = conv2D(layer1, 'd_conv2')						#[-1, 14, 14, 256]
	layer2 = batch_norm(layer2, 'd_conv2', is_training)		#[-1, 14, 14, 256]
	layer2 = activation(layer2, 'leaky')					#[-1, 14, 14, 256]
	layer2 = pooling(layer2, 'avg') 						#[-1, 7, 7, 256]

	# Reshape
	reshaped = reshape_tensor(layer2, [-1, 7*7*256]) 		#[-1, 7*7*256]

	# Dense/FC
	layer3 = dense(reshaped, 'd_dense1')					#[-1, 1024]
	layer3 = batch_norm(layer3, 'd_dense1', is_training)					#[-1, 1024]
	layer3 = activation(layer3, 'leaky')					#[-1, 1024]

	# Dense/FC
	layer4 = dense(layer3, 'd_dense2')						#[-1, num_class]
	layer4 = batch_norm(layer4, 'd_dense2', is_training)					#[-1, num_class]
	layer4 = activation(layer4, 'leaky')					#[-1, num_class]

	return layer4

def prepare_G_input(z_dim, cls):
	return

def train_model(epoches):
	graph = tf.Graph()
	with graph.as_default():
		
