import numpy as np
import tensorflow as tf
from ops import *
import time
import scipy.stats
from prepare_mnist import *
from prepare_cifar import *
from six.moves import cPickle as pickle

def generate_Z(batch_size, z_dim): 
	# sample from Gaussian normal
	lower = -1.
	upper = 1.
	mu = 0.
	sigma = 0.5
	Z = scipy.stats.truncnorm.rvs(
          (lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=[batch_size, z_dim])
	Z = Z.astype(np.float32)
	return Z

def embed(cls, Z):
	# Perform simple MLP to achieve cls and noise joint latent vector
	cls_embed = dense(cls, "embeding")
	assert(cls_embed.get_shape().as_list()==Z.get_shape().as_list())
	#cZ_joint = tf.multiply(cls_embed, Z) for TensorFlow 1.0
	cZ_joint = tf.mul(cls_embed, Z)
	return cZ_joint

def generator(cZ, is_training): # cZ (class and noise joint) latent vector: [batch_size, z_dim]
	# project cZ into dense layer1 - cZ - [-1, z_dim]
	layer1 = dense(cZ, "g_dense1")			#[-1, 4*4*384]
	# Batch Norm
	#layer1 = batch_norm(layer1, 'g_dense1', is_training) #[-1, 4*4*384]
	# Activation
	layer1 = activation(layer1, 'leaky') 	#[-1, 4*4*384]

	# reshape 
	reshaped = reshape_tensor(layer1, [-1, 4, 4, 384]) #[-1, 4, 4, 384]

	# Deconv 2D
	batch_size = reshaped.get_shape().as_list()[0]
	layer2 = deconv2D(reshaped, [batch_size, 8, 8, 192], 'g_deconv1') 	#[-1, 8, 8, 192]
	# Batch Norm
	layer2 = batch_norm(layer2, 'g_deconv1', is_training) 	#[-1, 8, 8, 192]
	# Activation
	layer2 = activation(layer2, 'leaky')		#[-1, 8, 8, 192]

	# Deconv 2D
	layer3 = deconv2D(layer2, [batch_size, 16, 16, 96], 'g_deconv2')	#[-1, 16, 16, 96]
	layer3 = batch_norm(layer3, 'g_deconv2', is_training)
	layer3 = activation(layer3, 'leaky')
 
	# Deconv 2D
	layer4 = deconv2D(layer3, [batch_size, 32, 32, 3], 'g_deconv3')		#[-1, 32, 32, 3]
	# output deconv layer - No batch norm
	layer4 = activation(layer4, 'tanh') # recommend 'tanh' output

	return layer4


def discriminator(X, is_training): # X - input image: [batch_size, 32, 32, 3]
	# Conv2D - X: [-1, 32, 32, 3]
	layer1 = conv2D(X, 'd_conv1') 								#[-1, 32, 32, 32]
	#layer1 = batch_norm(layer1, 'd_conv1', is_training)		#[-1, 32, 32, 32]
	layer1 = activation(layer1, 'leaky')						#[-1, 32, 32, 32]
	layer1 = pooling(layer1, 'avg') 							#[-1, 16, 16, 32]
	

	# Conv2D
	layer2 = conv2D(layer1, 'd_conv2')						#[-1, 16, 16, 64]
	layer2 = batch_norm(layer2, 'd_conv2', is_training)		#[-1, 16, 16, 64]
	layer2 = activation(layer2, 'leaky')					#[-1, 16, 16, 64]
	layer2 = pooling(layer2, 'avg') 						#[-1, 8, 8, 64]

	# Conv2D
	layer3 = conv2D(layer2, 'd_conv3')						#[-1, 8, 8, 128]
	layer3 = batch_norm(layer3, 'd_conv3', is_training)		#[-1, 8, 8, 128]
	layer3 = activation(layer3, 'leaky')					#[-1, 8, 8, 128]
	layer3 = pooling(layer3, 'avg') 						#[-1, 4, 4, 128]

	# Reshape
	reshaped = reshape_tensor(layer3, [-1, 4*4*128]) 		#[-1, 4*4*128]

	# Conv2D
	#layer4 = dense(reshaped, 'd_conv3')						#[-1, 1024]
	#layer4 = batch_norm(layer4, 'd_dense1', is_training)	#[-1, 1024]
	#layer4 = activation(layer4, 'leaky')					#[-1, 1024]

	# Dense/FC
	dense_cls = dense(reshaped, 'd_dense_cls')				#[-1, num_class]
	#layer4 = batch_norm(layer4, 'd_dense2', is_training)	#[-1, num_class]
	output_cls = activation(dense_cls, 'softmax')			#[-1, num_class]

	dense_src = dense(reshaped, 'd_dense_src')				#[-1, 1]
	output_src = activation(dense_src, 'sigmoid')			#[-1, 1]

	return [output_cls, output_src]



def train_model(Xr, yr, epoches, learning_rate):
	batch_size = 64
	real_src_np = np.ones([batch_size, 1], dtype=np.float32)
	fake_src_np = np.zeros([batch_size, 1], dtype=np.float32)

	# latent Z shape
	z_dim = 110
	num_class = 10

	G_wt_shapes = [ ['embeding', [num_class, z_dim]	, z_dim, 	False],
					['g_dense1', [z_dim, 4*4*384]	, 4*4*384,	False],
					['g_deconv1',[5, 5, 192, 384]	, 192, 		True],
					['g_deconv2',[5, 5, 96, 192]	, 96, 		True],
					['g_deconv3', [5, 5, 3, 96]		, 3, 		False] ]

	D_wt_shapes = [['d_conv1', [3, 3, 3, 32]	, 32, False],
				   ['d_conv2', [3, 3, 32, 64]	, 64, True],
				   ['d_conv3', [3, 3, 64, 128]   , 128, True],
				   #['d_conv4', [3, 3, 64, 128]	, 128, True],
				   ['d_dense_src', [4*4*128, 1]	, 1, False],# fake or real - sigmoid
				   ['d_dense_cls', [4*4*128, num_class], num_class, False]] # cls - softmax

	graph = tf.Graph()
	with graph.as_default():
		tf_z = tf.placeholder(tf.float32, shape=[batch_size, z_dim])
		#d_f = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
		real_images = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])
		real_src = tf.constant(real_src_np)
		fake_src = tf.constant(fake_src_np)
		true_cls = tf.placeholder(tf.float32, shape=[batch_size, num_class])

		is_train = tf.placeholder(tf.bool)

		# Initialize weights and biases within scopes 
		with tf.variable_scope("G"):
			for g_init in G_wt_shapes:
				initialize_variables(g_init[0], g_init[1], g_init[2], g_init[3])

		with tf.variable_scope("D"):
			for d_init in D_wt_shapes:
				initialize_variables(d_init[0], d_init[1], d_init[2], d_init[3])

		# Feed data forward into the network
		with tf.variable_scope("G", reuse=True):
			cZ = embed(true_cls, tf_z)
		with tf.variable_scope("G", reuse=True):
			fake_images = generator(cZ, is_training=is_train)

		with tf.variable_scope("D", reuse=True):	  
			[fk_cls, fk_src] = discriminator(fake_images, is_training=is_train)
		with tf.variable_scope("D", reuse=True):
			[rl_cls, rl_src] = discriminator(real_images, is_training=is_train)

		Lsrc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(rl_src, real_src)) + \
			   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fk_src, fake_src))
		Lcls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fk_cls, true_cls)) + \
			   tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(rl_cls, true_cls))

		rl_src_pred = tf.nn.sigmoid(rl_src)
		fk_src_pred = tf.nn.sigmoid(fk_src)
		rl_cls_pred = tf.nn.softmax(rl_cls)
		fk_cls_pred = tf.nn.softmax(fk_cls)

		Dloss = Lcls + Lsrc
		Gloss = Lcls - Lsrc
		# manage gradient update for the Discriminator
		#optimizer = tf.train.AdamOptimizer(learning_rate)
		#grads_and_vars = optimizer.compute_gradients(Dloss, \
			#tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D'))
		#optimizer.apply_gradients(grads_and_vars)

		g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(Gloss, 
			var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))
		d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(Dloss, 
			var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D'))

	Gloss_rcd, Dloss_rcd = np.zeros(epoches), np.zeros(epoches)
	fkimg_rcd = []
	src_pred = {'real_in': np.zeros(epoches), 'fake_in': np.zeros(epoches)}
	cls_pred = {'real_in': np.zeros(epoches), 'fake_in': np.zeros(epoches)}

	with tf.Session(graph=graph) as sess:
		#sess.run(tf.global_variables_initializer())
		tf.initialize_all_variables().run()
		print('Initialized')
		for epoch in range(epoches):
			t = time.time()
			offset = (epoch * batch_size) % (Xr.shape[0] - batch_size)
			batch_X = Xr[offset:(offset+batch_size), :, :, :]
			batch_y = yr[offset:(offset+batch_size), :]
			batch_Z = generate_Z(batch_size, z_dim)
			feed_dict = {real_images: batch_X, true_cls: batch_y, tf_z: batch_Z, is_train:True}
			_, _, Gls, Dls, fk_imgs, rl_pred, fk_pred, cls_pred_rlin, cls_pred_fkin = sess.run([g_optimizer, d_optimizer, \
				Gloss, Dloss, fake_images, rl_src_pred, fk_src_pred, rl_cls_pred, fk_cls_pred], feed_dict=feed_dict)
			#_, Gls, fk_imgs = sess.run([g_optimizer, Gloss, fake_images], feed_dict=feed_dict)
			#_, Dls			= sess.run([d_optimizer, Dloss], feed_dict=feed_dict)
			tm = time.time()-t
			Gloss_rcd[epoch] = Gls
			Dloss_rcd[epoch] = Dls
			rl_pred_acc, fk_pred_acc = compute_src_pred(rl_pred, fk_pred)
			rli_cls_acc, fki_cls_acc = compute_cls_pred(cls_pred_rlin, cls_pred_fkin, batch_y)
			src_pred['real_in'][epoch] = rl_pred_acc
			src_pred['fake_in'][epoch] = fk_pred_acc
			cls_pred['real_in'][epoch] = rli_cls_acc
			cls_pred['fake_in'][epoch] = fki_cls_acc

			if epoch%1000==0:
				d = {'Xreal': batch_X, 'yreal': batch_y, 'generated': fk_imgs}
				fkimg_rcd.append(d)
			print("Epoch: %d\tGloss: %.4f\tDloss: %.4f\trSrcPredAcc: %.2f%%\tfSrcPredAcc: %.2f%%\trClsPredAcc:%.2f%%\tfClsPredAcc:%.2f%%\tTime cost: %.2f"
				 %(epoch, Gls, Dls, rl_pred_acc*100, fk_pred_acc*100, rli_cls_acc*100, fki_cls_acc*100, tm))
		return Gloss_rcd, Dloss_rcd, fkimg_rcd, src_pred, cls_pred

def mnist_trial():
	mnist_fn = '/Users/Zhongyu/Documents/projects/kaggle/mnist/train.csv'
	#mnist_fn = '/home/paperspace/Documents/train.csv'
	Xr, yr = prepare_mnist2(mnist_fn)
	return Xr, yr

def cifar_trial():
	cifar_fn = "/Users/Zhongyu/Documents/projects/CNNplayground/cifar10/data/"
	#cifar_fn = '/home/paperspace/Documents/cifar_data/'
	datalist = prepare_cifar10_input(cifar_fn)
	Xr, yr = datalist[0], datalist[1]
	return Xr, yr

if __name__=='__main__':
	Xr, yr = cifar_trial()
	Grcd, Drcd, img_rcd, src_pred, cls_pred = train_model(Xr, yr, 5, 0.0002)
	rcd = {'Grcd':Grcd, 'Drcd':Drcd, 'img_rcd': img_rcd, 'src_pred': src_pred, 'cls_pred': cls_pred}
	with open('rcd0', 'w') as fh:
		pickle.dump(rcd, fh)


