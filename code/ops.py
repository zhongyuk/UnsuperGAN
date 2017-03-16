import numpy as np
import tensorflow as tf

def initialize_variables(scope_name, wt_shape, bi_shape, bn_bool,\
	initializer=tf.truncated_normal_initializer(stddev=.013)):
	with tf.variable_scope(scope_name) as scp:
		wt = tf.get_variable("wt", wt_shape, initializer=initializer)
		bi = tf.get_variable("bi", bi_shape, initializer=tf.constant_initializer(1.0))
		if bn_bool:
			gamma = tf.get_variable("gamma", bi_shape, initializer=tf.constant_initializer(1.0))
			beta = tf.get_variable("beta", bi_shape, initializer=tf.constant_initializer(0.0))
			moving_avg = tf.get_variable("moving_mean", bi_shape, initializer=tf.constant_initializer(0.0), trainable=False)
			moving_var = tf.get_variable("moving_variance", bi_shape, initializer=tf.constant_initializer(1.0), trainable=False)
		scp.reuse_variables()


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
	#tfleak = tf.constant(leak, dtype=tf.float32)
	#return tf.maximum(input, tf.scalar_mul(tfleak, input))
	return tf.maximum(input, leak*input)

def activation(input, type):
	if type == 'leaky':
		return lrelu(input)
	elif type == 'tanh':
		return tf.tanh(input)
	elif type == 'sigmoid':
		return tf.sigmoid(input)
	elif type == 'softmax':
		return tf.nn.softmax(input)
	elif type == 'relu':
		return tf.nn.relu(input)

def conv2D(input, scope_name, stride=1):
	with tf.variable_scope(scope_name, reuse=True):
		wt = tf.get_variable("wt")
		bi = tf.get_variable("bi")
		output = tf.nn.conv2d(input, wt, [1, stride, stride, 1], padding='SAME')
		output = output + bi
	return output

def batch_norm(input, scope_name, train, decay=0.9):
	output = tf.contrib.layers.batch_norm(input, decay=decay, is_training=train,
		center=True, scale=True, updates_collections=None, scope=scope_name, reuse=True)
	return output

def pooling(input, method, kernel=2, stride=2):
	if method=='max':
		return tf.nn.max_pool(input, [1,kernel,kernel,1], [1,stride,stride,1], padding='SAME')
	elif method=='avg':
		return tf.nn.avg_pool(input, [1,kernel,kernel,1], [1,stride,stride,1], padding='SAME')

def deconv2D(input, output_shape, scope_name, stride=2):
	with tf.variable_scope(scope_name, reuse=True):
		wt = tf.get_variable("wt")
		bi = tf.get_variable("bi")
		output = tf.nn.conv2d_transpose(input, wt, output_shape, [1, stride, stride, 1])
		output = output + bi
	return output

def compute_src_pred(rl_pred, fk_pred):
	batch_size = rl_pred.shape[0]
	rl_pred_acc = np.round(rl_pred).sum()/float(batch_size)
	fk_pred_acc = 1- np.round(fk_pred).sum()/float(batch_size)
	return rl_pred_acc, fk_pred_acc

def compute_cls_pred(rl_input, fk_input, labels):
	batch_size = rl_input.shape[0]
	rl_input_acc = np.sum(np.argmax(rl_input, 1) == np.argmax(labels, 1))/float(batch_size)
	fk_input_acc = np.sum(np.argmax(fk_input, 1) == np.argmax(labels, 1))/float(batch_size)
	return rl_input_acc, fk_input_acc







