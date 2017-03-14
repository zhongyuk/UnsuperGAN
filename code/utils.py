import numpy as np
import tensorflow as tf

def initialize_variables(scope_name, shape, initializer, bn_bool):
	with tf.variable_scope(scope_name) as scp:
		wt = tf.get_variable("wt", shape, initializer=initializer)
		bi = tf.get_variable("bi", shape[-1], initializer=tf.constant_initializer(1.0))
		if bn_bool:
			gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0))
			beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0))
			moving_avg = tf.get_variable("moving_mean", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
			moving_var = tf.get_variable("moving_variance", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
		scope.reuse_variables()


def dense(input, scope_name):
	with tf.variable_scope(scope_name, reuse=True):
		wt = tf.get_variable("wt")
		bi = tf.get_variable("bi")
		output = tf.matmul(input, wt) + bi
	return output

def reshape_tensor(input, target_shape):
	return tf.reshape(input, target_shape)

def upsample2D(input, xrepeat, yrepeat):
	repeat_input = tf.tile(input, xrepeat*yrepeat)
	input_shape = input.get_shape().as_list()
	newx, newy = input_shape[1], input_shape[2]
	new_shape = [input_shape[0], newx, newy, input_shape[3]]
	output = tf.reshape(input, new_shape)
	return output

def lrelu(input, leak=0.2):
	return tf.maximum(input, leak*x)

def activation(input, type):
	if type == 'leaky':
		return lrelu(input)
	elif type == 'tanh':
		return tf.tanh(input)
	elif type == 'sigmoid':
		return tf.sigmoid(input)
	elif type == 'relu':
		return tf.nn.relu(input)

def conv2D(input, scope_name, stride=2):
	with tf.variable_scope(scope_name, reuse=True):
		wt = tf.get_variable("wt")
		bi = tf.get_variable("bi")
		output = tf.nn.conv2d(input, wt, [1, stride, stride, 1])
		output = output + bi
	return output

def batch_norm(input, scope_name, train, decay=0.9):
	output = tf.contrib.layers.batch_norm(input, decay=decay, is_training=train,
		center=True, scale=True, updates_collections=None, scope=scope_name, reuse=True)

def pooling(input, method, kernel=2, stride=2):
	if method=='max':
		return tf.nn.max_pool(input, [1,kernel,kernel,1], [1,stride,stride,1])
	elif method=='avg':
		return tf.nn.avg_pool(x, [1,kernel,kernel,1], [1,stride,stride,1])

def deconv2D(input, scope_name, stride=2):
	with tf.variable_scope(scope_name, reuse=True):
		wt = tf.get_variable("wt")
		bi = tf.get_variable("bi")
		output = tf.nn.conv2d_transpose(input, wt, [1, stride, stride, 1])
		output = output + bi
	return output







