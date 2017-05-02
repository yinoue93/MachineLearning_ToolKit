import matplotlib.pyplot as plt
import numpy as np

def setupUtils():
	plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
	plt.rcParams['image.interpolation'] = 'nearest'
	plt.rcParams['image.cmap'] = 'gray'

def time_function(f, *args):
	"""
	Call a function f with args and return the time (in seconds) that it took to execute.
	"""
	import time
	tic = time.time()
	f(*args)
	toc = time.time()
	return toc - tic

def plot_1d_crossvalidation(var_accuracies):
	"""
	plot the crossvalidation results

	@type	var_accuracies	:	dict of lists
	@param	var_accuracies	:	keys should be values crossvalidated, each list
								should be a list of accuracies for that crossvalidated
								param setting
	"""
	var_choices = var_accuracies.items()
	for var in var_choices:
		accuracies = var_accuracies[k]
		plt.scatter([k] * len(accuracies), accuracies)

	# plot the trend line with error bars that correspond to standard deviation
	accuracies_mean = np.array([np.mean(v) for k,v in sorted(var_choices)])
	accuracies_std = np.array([np.std(v) for k,v in sorted(var_choices)])
	plt.errorbar(var_choices, accuracies_mean, yerr=accuracies_std)
	plt.title('Cross-validation on variables')
	plt.xlabel('k')
	plt.ylabel('Cross-validation accuracy')
	plt.show()

def weights_visualization(weights, w_shape=None, names=None):
	"""
	Visualize the learned weights, allows for labeling weights
	More suitable for small number of weights

	@type	weights	:	list of numpy arrays
	@param	weights	:	learned weights
	@type	w_shape	:	tuple
	@param	w_shape	:	shape of each weight, None if it is already reshaped correctly.
	@type	names	:	list of strings
	@param	names	:	names of the weights
	"""

	for i,w in enumerate(weights):
		w = w.reshape(w_shape)
		w_min, w_max = np.min(w), np.max(w)
		plt.subplot(2, 5, i + 1)
		  
		# Rescale the weights to be between 0 and 255
		wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
		plt.imshow(wimg.astype('uint8'))
		plt.axis('off')
		plt.title(names[i])

def show_weights(weights, names=None):
	"""
	Visualize the learned weights
	More suitable for many weights

	@type	weights	:	numpy array
	@param	weights	:	learned weights of shape (N,H,W,C)
	@type	names	:	list of strings
	@param	names	:	names of the weights 
	"""
	plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
	plt.gca().axis('off')
	plt.show()

#-------------------------to be implemented (application specific)-------------------------
def load_data(dev=False):
	X_train, y_train, X_test, y_test = [],[],[],[]

	if dev:
		mask = np.random.choice(num_training, num_dev, replace=False)
		X_train = X_train[mask]
		y_train = y_train[mask]


	return X_train, y_train, X_test, y_test