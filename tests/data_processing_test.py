import os
import inspect
import sys

this_folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_folder = os.path.dirname(this_folder)
sys.path.append(parent_folder)

import data_processing as dp
import numpy as np
import unittest
import csv

class DataProcessingTests(unittest.TestCase):
	def test_ensure_unit_variance(self):
		data = np.linspace(1,50).reshape((10,5))
		stds = [np.std(column) for column in data.T]
		uv_data = dp.ensure_unit_variance(data, [0,2,3])
		
		data[:,0] = (data[:,0] - np.mean(data[:,0])) / stds[0]
		data[:,2] = (data[:,2] - np.mean(data[:,2])) / stds[2]
		data[:,3] = (data[:,3] - np.mean(data[:,3])) / stds[3]
		
		for target, actual in zip(data.T, uv_data.T):
			self.assertAlmostEqual(np.mean(actual), np.mean(target))
			self.assertAlmostEqual(np.std(actual), np.std(target))

	def test_remove_outliers(self):
		dataX = np.linspace(-122.535908,-122.347306,100)
		dataY = np.linspace(37.696850,37.839763, 100)
		data = [(x,y) for x,y in zip(dataX, dataY)]
		data.extend( [(10,10), (200,200)] )
		data = np.asarray(data)
		
		ro_data = dp.remove_outliers(data, 0, 1)
		ro_data = [point for point in ro_data]
		ro_data = np.asarray(ro_data)
		
		self.assertEqual(ro_data.shape[0], data.shape[0] - 2)
			
if __name__ == "__main__":
    unittest.main()