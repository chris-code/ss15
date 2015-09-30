import math
import numpy as np
import importer as im
import sklearn as sl
import sklearn.cross_validation as cv

class Network:
	def calc_eta(self, t):
		return 1 / math.sqrt(10000 + t)

	def sigmoidal(self, values):
		return 1 / (1 + np.exp(-values))
	
	def sigmoidal_derivative(self, values):
		s = self.sigmoidal(values)
		return (1 - s) * s
		
	def initialize(self, input_neuron_count, hidden_neuron_count, output_neuron_count):
		self.hidden_layer = np.empty((hidden_neuron_count))
		self.output_layer = np.empty((output_neuron_count))
		self.i2h_weights = np.random.normal(0, 1./100, size=(hidden_neuron_count, input_neuron_count))
		self.h2o_weights = np.random.normal(0, 1./100, size=(output_neuron_count, hidden_neuron_count))
	
	def predict(self, input_layer, get_u = False):
		u1 = np.dot(self.i2h_weights, input_layer)
		self.hidden_layer[...] = self.sigmoidal(u1)
		u2 = np.dot(self.h2o_weights, self.hidden_layer)
		self.output_layer[...] = self.sigmoidal(u2)
		
		if get_u: return np.argmax(self.output_layer), u1, u2
		else: return np.argmax(self.output_layer)
	
	def train(self, eta, label, input_layer):
		prediction, u1, u2 = self.predict(input_layer, True)
		
		delta2 = (self.output_layer - label) * (1 - self.output_layer) * self.output_layer
		derivative2 = np.outer(delta2, self.hidden_layer)
		
		delta1 = np.dot( delta2, np.dot( self.h2o_weights, np.diag(self.sigmoidal_derivative(u1)) ) )
		derivative1 = np.outer(delta1, self.sigmoidal(input_layer))
		
		self.h2o_weights[...] -= eta * derivative2
		self.i2h_weights[...] -= eta * derivative1
		
	def run(self, data_train, target_train):
		for iteration in range(1000000):
			# if iteration % 1000 == 0:
				# print(iteration)
			self.train(self.calc_eta(iteration), target_train[iteration % data_train.shape[0]],
				data_train[iteration % data_train.shape[0],:])

	def calculate_error(self, data, target):
		mistakes = 0
		for index in range(data.shape[0]):
			prediction = nn.predict(data[index, :])
			if prediction != target[index]:
				mistakes += 1
		return float(mistakes) / data.shape[0]
		
path = "../data/train.csv"
data = im.to_numpy_array(im.preprocess(im.vectorize(im.read(path, 30000))))
X = data[:,2:4]
Y = data[:,1]

data_train, data_test, target_train, target_test = cv.train_test_split(X, Y, test_size=0.33)

hh = [175,185,195,200,205,215]

for h in hh:
	nn = Network()
	nn.initialize(data_train.shape[1], h, 39)
	nn.run(data_train, target_train)
	error_train = nn.calculate_error(data_train, target_train)
	error_test = nn.calculate_error(data_test, target_test)
	print('Score on training set with h = {1}: {0}'.format(1. - error_train, h))
	print('Score on test set with h = {1}: {0}'.format(1. - error_test, h))