def cross_validation():
	num_folds = 5
	k_choices = [1, 3, 4, 5, 10, 12, 15, 20, 50, 100]

	X_train_folds = []
	y_train_folds = []

	"""
	Split up the training data into folds. After splitting, X_train_folds and    
	y_train_folds should each be lists of length num_folds, where                
	y_train_folds[i] is the label vector for the points in X_train_folds[i].
	"""

	X_train_folds = np.array_split(X_train, num_folds)
	y_train_folds = np.array_split(y_train, num_folds)

	"""
	A dictionary holding the accuracies for different values of k that we find
	when running cross-validation. After running cross-validation,
	k_to_accuracies[k] should be a list of length num_folds giving the different
	accuracy values that we found when using that value of k.
	"""
	k_to_accuracies = {}

	"""
	Perform k-fold cross validation to find the best value of k. For each        
	possible value of k, run the k-nearest-neighbor algorithm num_folds times,   
	where in each case you use all but one of the folds as training data and the 
	last fold as a validation set. Store the accuracies for all fold and all     
	values of k in the k_to_accuracies dictionary.                               
	"""

	for k in k_choices:
		k_to_accuracies[k] = []
		
		for fold in range(num_folds):
			# load appropriate data
			X_test_cv = []
			y_test_cv = []
			X_train_cv = []
			y_train_cv = []
			
			for i in range(num_folds):
				if i==fold:
					X_test_cv = X_train_folds[i]
					y_test_cv = y_train_folds[i]
				else:
					X_train_cv.append(X_train_folds[i])
					y_train_cv.append(y_train_folds[i])
			
			X_train_cv = np.vstack(X_train_cv)
			y_train_cv = np.hstack(y_train_cv)
			
			# ML
			classifier = myClassifier()
			classifier.train(X_train_cv, y_train_cv)
			
			dists = classifier.compute_distances_no_loops(X_test_cv)
			y_test_pred = classifier.predict_labels(dists, k=k)

			# Compute and print the fraction of correctly predicted examples
			num_correct = np.sum(y_test_pred == y_test_cv)
			accuracy = float(num_correct) / y_test_cv.size
			
			k_to_accuracies[k].append(accuracy)

	# Print out the computed accuracies
	for k in sorted(k_to_accuracies):
		for accuracy in k_to_accuracies[k]:
			print('k = %d, accuracy = %f' % (k, accuracy))

def hyperparamTuning():
	learning_rates = [2.5e-7, 4e-7]
	regularization_strengths = [2e3, 3.5e3, 5e3]

	"""
	results is dictionary mapping tuples of the form
	(learning_rate, regularization_strength) to tuples of the form
	(training_accuracy, validation_accuracy). The accuracy is simply the fraction
	of data points that are correctly classified.
	"""
	results = {}
	best_val = -1   # The highest validation accuracy that we have seen so far.
	best_svm = None # The LinearSVM object that achieved the highest validation rate.

	"""
	Write code that chooses the best hyperparameters by tuning on the validation 
	set. For each combination of hyperparameters, train a linear SVM on the      
	training set, compute its accuracy on the training and validation sets, and  
	store these numbers in the results dictionary. In addition, store the best   
	validation accuracy in best_val and the best model in best_model.
	"""

	count = 0
	total_settings = len(learning_rates) * len(regularization_strengths)
	for lr in learning_rates:
		for regular in regularization_strengths:
			count += 1
			print("Running %d of %d hyperparameters..." %(count,total_settings))
			
			model = myClassifier()
			model.train(X_train, y_train, lr, regular, num_iters=5000)
			
			y_train_pred = model.predict(X_train)
			train_accuracy = np.mean(y_train == y_train_pred)
			
			y_dev_pred = model.predict(X_val)
			val_accuracy = np.mean(y_val == y_dev_pred)
			
			results[(lr, regular)] = (train_accuracy, val_accuracy)
			if val_accuracy > best_val:
				best_model = model
				best_val = val_accuracy
			
			print('lr %e reg %e train accuracy: %f val accuracy: %f' % \
								(lr, regular, train_accuracy, val_accuracy))
		
	print('best validation accuracy achieved during cross-validation: %f' % best_val)


def hyperparamTuning_NN():
	# Hyperparameter search

	import itertools
	param_key = ['param1', 'param2', 'param3']

	param1 = []
	param2 = []
	param3 = []

	ps = [param1, param2, param3]
	params = list(itertools.product(*ps))

	count = 0
	best_val = 0
	best_param = None
	for param in params:
		# build a new computational graph
		tf.reset_default_graph()
		X = tf.placeholder(tf.float32, [None, 32, 32, 3])
		y = tf.placeholder(tf.int64, [None])
		is_training = tf.placeholder(tf.bool)
		
		y_out = my_model(X, y, is_training, param)

		# define our loss
		with tf.variable_scope('vars') as scope:
			scope.reuse_variables()
			regularizer = tf.nn.l2_loss(tf.get_variable('Wconv1')) + tf.nn.l2_loss(tf.get_variable('Wconv2')) \
							 + tf.nn.l2_loss(tf.get_variable('W1')) + tf.nn.l2_loss(tf.get_variable('W2'))

		total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,10), logits=y_out)
		mean_loss = tf.reduce_mean(total_loss + param[-1]*regularizer)

		# define our optimizer
		optimizer = tf.train.RMSPropOptimizer(param[-2])
		train_step = optimizer.minimize(mean_loss)

		count += 1
		print('Running #%d of %d runs...' %(count, len(params)))
		param_str = ''
		for i in range(len(param)):
			param_str += param_key[i] + ':' + str(param[i]) + ', '
		print(param_str[:-2])
		
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		_,train_acc = run_model(sess,y_out,mean_loss,X_train,y_train,1,64,1000,train_step,plot_losses=False,quiet=True)
		_,val_acc = run_model(sess,y_out,mean_loss,X_val,y_val,1,64,quiet=True)
		
		print('train acc: %f, val acc: %f' %(train_acc, val_acc))
		if best_val<val_acc:
			best_val = val_acc
			best_param = param
			
		print('')

	print('Best validation accuracy: %f' % best_val)
	print(best_param)