import os
import inspect
import sys
this_folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_folder = os.path.dirname(this_folder)
sys.path.append(parent_folder)

import unittest
import csv
import numpy as np
import nearest_neighbors

class Test_Distance_In_Mod(unittest.TestCase):
	def setUp(self):
		self.integer_test_values = [(4, 5, 10, 1), (1, 9, 10, 2), (-2, 2, 10, 4), (-4, 4, 10, 2)]
		self.float_test_values = [(4.0, 5.0, 10.0, 1.0), (1.0, 9.0, 10.0, 2.0), (-2.0, 2.0, 10.0, 4.0), (-4.0, 4.0, 10.0, 2.0)]
	
	def test_integer(self):
		for test in self.integer_test_values:
			a = nearest_neighbors.distance_in_mod(test[0], test[1], test[2])
			self.assertEqual(a, test[3])
	
	def test_integer_reverse(self):
		for test in self.integer_test_values:
			a = nearest_neighbors.distance_in_mod(test[1], test[0], test[2])
			self.assertEqual(a, test[3])
	
	def test_integer_sign(self):
		for test in self.integer_test_values:
			a = nearest_neighbors.distance_in_mod(test[0], test[1], test[2])
			self.assertGreater(a, 0)

	def test_float(self):
		for test in self.float_test_values:
			a = nearest_neighbors.distance_in_mod(test[0], test[1], test[2])
			self.assertAlmostEqual(a, test[3])
	
	def test_float_reverse(self):
		for test in self.float_test_values:
			a = nearest_neighbors.distance_in_mod(test[1], test[0], test[2])
			self.assertAlmostEqual(a, test[3])
	
	def test_float_sign(self):
		for test in self.float_test_values:
			a = nearest_neighbors.distance_in_mod(test[0], test[1], test[2])
			self.assertGreater(a, 0.0)

class Test_Distance_Function(unittest.TestCase):
	def setUp(self):
		data_path = 'src/tests/data/distances.csv'
		
		with open(data_path, 'r') as file:
			descriptions = file.readline().split(',') # The first line of the .csv file. We discard this.
			csv_reader = csv.reader(file)
			
			self.data_points = []
			for index, data_point in enumerate(csv_reader):
				if index % 2 == 0:
					d1 = data_point[:-4]
					for i in range(5):
						d1[i] = float(d1[i])
					for i in range(5, 8):
						d1[i] = int(d1[i])
				else:
					d2 = data_point[:-4]
					for i in range(5):
						d2[i] = float(d2[i])
					for i in range(5, 8):
						d2[i] = int(d2[i])
					
					modulae = (float(data_point[-4]), float(data_point[-3]), float(data_point[-2]))
					true_distance = float(data_point[-1])
					
					self.data_points.append( (d1, d2, modulae, true_distance) )
	
	def test_distance(self):
		for d1, d2, modulae, true_distance in self.data_points:
			calculated_distance = nearest_neighbors.distance_function(d1, d2, modulae=modulae)
			self.assertAlmostEqual(calculated_distance, true_distance)

class Test_train(unittest.TestCase):
	# def test_valid(self):
		# Generate zero mean, unit variance data
		# loc_train = np.random.normal(0, 1, size=(100, 8)) # 100 samples, and the required number of features for distance function
		# loc_test = np.random.normal(0, 1, size=(100, 8))
		# crime_ids_train = np.random.random_integers(0, 38, size=(100)) 
		# crime_ids_test = np.random.random_integers(0, 38, size=(100))
		# neighbor_counts = [1]
		
		# nearest_neighbors.train(loc_train, loc_test, crime_ids_train, crime_ids_test, neighbor_counts)
	def test_umatched_size(self):
		loc_train = np.random.normal(0, 1, size=(100, 8)) # 100 samples, and the required number of features for distance function
		loc_test = np.random.normal(0, 1, size=(100, 8))
		crime_ids_train = np.random.random_integers(0, 38, size=(50)) # Only 50 samples
		crime_ids_test = np.random.random_integers(0, 38, size=(100))
		neighbor_counts = [1]
		
		f = nearest_neighbors.train
		self.assertRaises(ValueError, f, loc_train, loc_test, crime_ids_train, crime_ids_test, neighbor_counts)

if __name__ == '__main__':
	unittest.main()
















