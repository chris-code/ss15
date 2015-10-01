import nn
import numpy as np
import unittest

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
	
if __name__ == "__main__":
    unittest.main()
	