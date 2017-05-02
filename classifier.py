from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class myGeneralClassifier(object):
	"""
	Example classifier implementation.
	"""

	def __init__(self):
		"""
		Initialize the model. Store characterizing parameters in self.params dictionary.
		"""
		self.params = {}

	def loss(self, X, y=None, reg=0.0):
		"""
		Compute the loss and gradients.
		"""
		# Unpack variables from the params dictionary

		# Compute the forward pass
		scores = None

		# If the targets are not given then jump out, we're done
		if y is None:
			return scores

		# Compute the loss
		loss = None
		
		# Backward pass: compute gradients
		grads = {}
		
		return loss, grads

	def train(self, X, y, X_val, y_val, verbose=False):
		"""
		Train this neural network using stochastic gradient descent.

		Inputs:
		- X: A numpy array of shape (N, D) giving training data.
		- y: A numpy array of shape (N,) giving training labels; y[i] = c means that
			X[i] has label c, where 0 <= c < C.
		- X_val: A numpy array of shape (N_val, D) giving validation data.
		- y_val: A numpy array of shape (N_val,) giving validation labels.
		- verbose: boolean; if true print progress during optimization.
		"""
		num_train = X.shape[0]
		iterations_per_epoch = max(num_train / batch_size, 1)

		# Use SGD to optimize the parameters in self.model
		loss_history = []
		train_acc_history = []
		val_acc_history = []

		for it in xrange(num_iters):
			X_batch = None
			y_batch = None

			# Create a random minibatch of training data and labels, storing
			# them in X_batch and y_batch respectively.                             
			
			indx = np.random.choice(num_train, size=batch_size)
			X_batch = X[indx, :]
			y_batch = y[indx]
		
			# Compute loss and gradients using the current minibatch
			loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
			loss_history.append(loss)

			"""
			Use the gradients in the grads dictionary to update the         
			parameters of the network (stored in the dictionary self.params).
			"""                    
			
			self.params['W1'] -= learning_rate * grads['W1']
			self.params['W2'] -= learning_rate * grads['W2']
			self.params['b1'] -= learning_rate * grads['b1']
			self.params['b2'] -= learning_rate * grads['b2']

			if verbose and it % 100 == 0:
				print('iteration %d / %d: loss %f' % (it, num_iters, loss))

			# Every epoch, check train and val accuracy and decay learning rate.
			if it % iterations_per_epoch == 0:
				# Check accuracy
				train_acc = (self.predict(X_batch) == y_batch).mean()
				val_acc = (self.predict(X_val) == y_val).mean()
				train_acc_history.append(train_acc)
				val_acc_history.append(val_acc)

				# Decay learning rate
				learning_rate *= learning_rate_decay

		return {
			'loss_history': loss_history,
			'train_acc_history': train_acc_history,
			'val_acc_history': val_acc_history,
		}

	def predict(self, X):
		"""
		Inputs:
		- X: A numpy array of shape (N, D) giving N D-dimensional data points to
			classify.

		Returns:
		- y_pred: A numpy array of shape (N,) giving predicted labels for each of
			the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
			to have class c, where 0 <= c < C.
		"""
		y_pred = None
		
		return y_pred

def myTFmodel(X, y, is_training, param):	
	# expand param
	param1, param2, param3 = param
	
	# define the graph
	with tf.variable_scope('vars') as scope:
		Wconv1 = tf.get_variable("Wconv1", shape=[filter_sz1, filter_sz1, 3, num_filters1])
		bconv1 = tf.get_variable("bconv1", shape=[num_filters1])
		c1 = tf.nn.conv2d(X, Wconv1, strides=[1,cnn_stride_sz1,cnn_stride_sz1,1], padding='VALID') + bconv1
		a1 = tf.nn.relu(c1)

		bn_gamma = tf.get_variable("bn_gamma", shape=[after_conv.get_shape().as_list()[-1]])
		bn_beta = tf.get_variable("bn_beta", shape=[after_conv.get_shape().as_list()[-1]])
		batch_mean,batch_var = tf.nn.moments(after_conv, axes=[0, 1, 2])
		bn1 = tf.nn.batch_normalization(after_conv, batch_mean, batch_var, bn_beta, bn_gamma, 1e-3)

		m1 = tf.nn.max_pool(bn1, ksize=[1, max_pool_stride_sz, max_pool_stride_sz, 1], 
							strides=[1, max_pool_stride_sz, max_pool_stride_sz, 1], padding='VALID')
		
		aff1_inputs = 1
		for i in m1.get_shape().as_list()[1:]:
			aff1_inputs *= i
		
		W1 = tf.get_variable("W1", shape=[aff1_inputs, hidden_sz1])
		b1 = tf.get_variable("b1", shape=[hidden_sz1])
		m1_flat = tf.reshape(m1,[-1, aff1_inputs])
		aff1 = tf.matmul(m1_flat, W1) + b1
		
		W2 = tf.get_variable("W2", shape=[hidden_sz1, num_classes])
		b2 = tf.get_variable("b2", shape=[num_classes])
		a2 = tf.nn.relu(aff1)
	
	y_out = tf.matmul(a2, W2) + b2
	
	return y_out

def trainBestTFModel():
	# train the best model
	best_param = (4, 128, 2, 4, 128, 1, 1, 512, 0.0005, 0.001)
	# best_param = (4, 256, 2, 2, 64, 1, 1, 512, 0.001, 0.001)
	# best_param = (4, 256, 2, 4, 64, 1, 1, 1024, 0.001, 0.001)

	# build a new computational graph
	tf.reset_default_graph()
	X = tf.placeholder(tf.float32, [None, 32, 32, 3])
	y = tf.placeholder(tf.int64, [None])
	is_training = tf.placeholder(tf.bool)

	y_out = my_model(X,y,is_training,best_param)

	# define our loss
	with tf.variable_scope('vars') as scope:
		scope.reuse_variables()
		regularizer = tf.nn.l2_loss(tf.get_variable('Wconv1')) + tf.nn.l2_loss(tf.get_variable('Wconv2')) \
					 + tf.nn.l2_loss(tf.get_variable('W1')) + tf.nn.l2_loss(tf.get_variable('W2'))
			
	total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,10), logits=y_out)
	mean_loss = tf.reduce_mean(total_loss + best_param[-1]*regularizer)

	# define our optimizer (also decay the learning rate)
	lr = tf.Variable(best_param[-2], trainable=False)
	optimizer = tf.train.RMSPropOptimizer(lr)
	train_step = optimizer.minimize(mean_loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	for i in range(10):
		_,train_acc = run_model(sess,y_out,mean_loss,X_train,y_train,1,64,1000,train_step,plot_losses=False,quiet=True)
		_,val_acc = run_model(sess,y_out,mean_loss,X_val,y_val,1,64,quiet=True)
		
		lr *= 0.5
		
		print('%d train acc: %f, val acc: %f' %(i, train_acc, val_acc))