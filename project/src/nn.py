import math
import numpy as np
import importer as im
import evaluation as ev
import data_processing as dp
import sklearn as sl
import sklearn.cross_validation as cv

class Network:
	def calc_eta(self, t):
		'''Returns the value of the learning rate for iteration 't'.'''
		return 1 / math.sqrt(10000 + t)

	def sigmoidal(self, values):
		'''Pointwise calculation of the activation function.'''
		return 1 / (1 + np.exp(-values))
	
	def sigmoidal_derivative(self, values):
		'''Returns the derivative of the activation function.'''
		s = self.sigmoidal(values)
		return (1 - s) * s
		
	def initialize(self, input_neuron_count, hidden_neuron_count, output_neuron_count):
		'''Initializes the hidden layer and the output layer with the given number of neurons.'''
		# initialize layers
		self.hidden_layer = np.empty((hidden_neuron_count))
		self.output_layer = np.empty((output_neuron_count))
		
		# initialize weights
		self.i2h_weights = np.random.normal(0, 1./100, size=(hidden_neuron_count, input_neuron_count))
		self.h2o_weights = np.random.normal(0, 1./100, size=(output_neuron_count, hidden_neuron_count))
	
	def predict(self, input_layer, get_u = False):
		'''Predicts the label for a given data point / for given input layer values.'''
		# calculate input into and output out of hidden layer
		u1 = np.dot(self.i2h_weights, input_layer)
		self.hidden_layer[...] = self.sigmoidal(u1)
		
		# calculate input into and output out of output layer
		u2 = np.dot(self.h2o_weights, self.hidden_layer)
		self.output_layer[...] = self.sigmoidal(u2)
		
		# return most likely label (and layer inputs if desired)
		if get_u: return np.argmax(self.output_layer), u1, u2
		else: return np.argmax(self.output_layer)
	
	def train(self, eta, label, input_layer):
		'''Trains the network via backpropagation.'''
		# predict label of the data point and get input values for the layers
		prediction, u1, u2 = self.predict(input_layer, True)

		# calculate derivative for the weights between hidden and output layer
		delta2 = (self.output_layer - label) * (1 - self.output_layer) * self.output_layer
		derivative2 = np.outer(delta2, self.hidden_layer)

		# calculate derivative for the weights between input and hidden layer
		delta1 = np.dot( delta2, np.dot( self.h2o_weights, np.diag(self.sigmoidal_derivative(u1)) ) )
		derivative1 = np.outer(delta1, self.sigmoidal(input_layer))

		# adapt weights
		self.h2o_weights[...] -= eta * derivative2
		self.i2h_weights[...] -= eta * derivative1
		
	def run(self, iterations, data_train, target_train):
		'''Runs the network with given training data for the given number of iterations.'''
		for iteration in range(iterations):
			if iteration % 100000 == 0:
				print iteration
			self.train(self.calc_eta(iteration), target_train[iteration % data_train.shape[0]],
				data_train[iteration % data_train.shape[0],:])

	def calculate_error(self, data, target):
		'''Calculates the error for given data and predicted labels.'''
		mistakes = 0
		for index in range(data.shape[0]):
			prediction = nn.predict(data[index, :])
			if prediction != target[index]:
				mistakes += 1
		return float(mistakes) / data.shape[0]

		
# load data
path = "../data/train.csv"
data = im.read_labeled(path, 3000)
data = dp.vectorize(data, 1, features=[('latitude', 7), ('longitude', 8), ('day', 0), ('day_of_week', 0), ('time', 0)])
crime_to_id_dict = data.next()
data = im.to_numpy_array(data)
data = dp.ensure_unit_variance(data, columns_to_normalize=(0,1,2,3,4))

# separate data in features and labels
Y = data[:,5].astype(int)
X = data[:,:5]

# split data in training data and test data
data_train, data_test, target_train, target_test = cv.train_test_split(X, Y, test_size=0.33)

# run network for a set of different hidden layer sizes
for h in [175,185,195,200,205,215]:
	nn = Network()
	nn.initialize(data_train.shape[1], h, 39)
	nn.run(1000000, data_train, target_train)
	error_train = nn.calculate_error(data_train, target_train)
	error_test = nn.calculate_error(data_test, target_test)
	print('Score on training set with h = {1}: {0}'.format(1. - error_train, h))
	print('Score on test set with h = {1}: {0}'.format(1. - error_test, h))