import os
import inspect
import sys

this_folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_folder = os.path.dirname(this_folder)
sys.path.append(parent_folder)

import nn
import numpy as np
import unittest
import csv


class NetworkTests(unittest.TestCase):
	def setUp(self):
		self.N = nn.Network()

	def test_calc_eta(self):
		self.assertAlmostEqual(self.N.calc_eta(30), 0.00998503366585)
		
	def test_sigmoidal(self):
		self.assertAlmostEqual(self.N.sigmoidal(-1), 0.26894142)
		self.assertAlmostEqual(self.N.sigmoidal(0), 0.5)
		self.assertAlmostEqual(self.N.sigmoidal(1), 0.73105858)

	def test_sigmoidal_derivative(self):
		self.assertAlmostEqual(self.N.sigmoidal_derivative(-1), 0.19661193)
		self.assertAlmostEqual(self.N.sigmoidal_derivative(0), 0.25)
		self.assertAlmostEqual(self.N.sigmoidal_derivative(1), 0.19661193)
	
	def test_initialize(self):
		self.N.initialize(5,10,3)
		self.assertEqual(self.N.hidden_layer.shape[0],10)
		self.assertEqual(self.N.output_layer.shape[0],3)
		self.assertEqual(self.N.i2h_weights.shape, (10,5))
		self.assertEqual(self.N.h2o_weights.shape, (3,10))
	
	def test_predict(self):
		self.N.initialize(5,10,3)
		self.N.i2h_weights = np.array(range(50)).reshape(10,5) / 100.
		self.N.h2o_weights = np.array(range(30)).reshape(3,10) / 100.
		self.assertEqual(self.N.predict(np.array([2,3,4,2,1])), 2)
		
	def test_train(self):
		self.N.initialize(5,10,3)
		self.N.i2h_weights = np.array(range(50)).reshape(10,5) / 100.
		self.N.h2o_weights = np.array(range(30)).reshape(3,10) / 100.
		self.N.train(self.N.calc_eta(30), 1, np.array([3,5,7,11,13]))
		
		for path in ["src/tests/data/train_test_i2h.csv", "src/tests/data/train_test_h2o.csv"]:
			with open(path, 'r') as file:
				csv_reader = csv.reader(file)
				data_list = []
				for data_point in csv_reader:	
					data_list.append(data_point)
				desired_weights = np.array(data_list)
			
			for i,row in enumerate(desired_weights):
				for j,element in enumerate(row):
					if path == "src/tests/data/train_test_i2h.csv":
						self.assertAlmostEqual(float(element), self.N.i2h_weights[i,j])
					else:
						self.assertAlmostEqual(float(element), self.N.h2o_weights[i,j])

	def test_calculate_error(self):
		self.N.initialize(5,10,3)
		self.N.i2h_weights = np.array(range(50)).reshape(10,5) / 100.
		self.N.h2o_weights = np.array(range(30)).reshape(3,10) / 100.
		self.N.train(self.N.calc_eta(30), 1, np.array([3,5,7,11,13]))
		data = np.array([[3,5,7,11,13],[4,6,8,10,12],[3,6,7,10,13]])
		target = np.array([1,0,2])
		self.assertAlmostEqual(self.N.calculate_error(data, target), 2./3.)
		
		
if __name__ == "__main__":
    unittest.main()
	
	# NN = nn.Network()
	# NN.initialize(5,10,3)
	# NN.i2h_weights = np.array(range(50)).reshape(10,5) / 100.
	# NN.h2o_weights = np.array(range(30)).reshape(3,10) / 100.
	# NN.train(NN.calc_eta(30), 1, np.array([3,5,7,11,13]))

	# data = np.array([[3,5,7,11,13],[4,6,8,10,12],[3,6,7,10,13]])
	# target = np.array([1,0,2])
	
	# print NN.calculate_error(data, target)
	
	
	# for path in ["tests/data/train_test_i2h.csv", "tests/data/train_test_h2o.csv"]:
		# with open(path, 'w') as file: # FIXME change 3
			# csv_writer = csv.writer(file, delimiter=',')
			
			# weights = ""
			# if path == "tests/data/train_test_i2h.csv":
				# weights = NN.i2h_weights
			# else:
				# weights = NN.h2o_weights
			
			# for line in weights:
				# row = line.tolist()
				# csv_writer.writerow( row )
	
	