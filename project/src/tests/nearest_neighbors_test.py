import unittest
import nearest_neighbors

class Test_Distance_In_Mod(unittest.TestCase):
	def setUp(self):
		self.integer_test_values = [(4, 5, 10, 1), (1, 9, 10, 1), (-2, 2, 10, 4), (-4, 4, 10, 3)]
		self.float_test_values = [(4.0, 5.0, 10.0, 1.0), (1.0, 9.0, 10.0, 1.0), (-2.0, 2.0, 10.0, 4.0), (-4.0, 4.0, 10.0, 3.0)]
	
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
