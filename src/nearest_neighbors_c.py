import math
import numpy as np

def _default_metric(a, b):
	return math.sqrt( ((a - b)**2).sum() )

class Nearest_Neighbor_Classifier:
	
	def __init__(self, n_neighbors, metric=_default_metric, metric_params={}):
		self.n_neighbors = n_neighbors
		#~ self.metric = metric
		#~ self.metric_params = metric_params
		self.metric = lambda a, b: metric(a, b, **metric_params)
		
		self.train_X = None
		self.train_y = None
	
	def fit(self, X, y):
		if X.shape[0] != y.size:
			raise ValueError('X has shape {0} but y has shape {1}'.format(X.shape, y.shape))
		
		if self.train_X is None:
			self.train_X = X
			self.train_y = y
		else:
			self.train_X = np.vstack( [self.train_X, X] )
			self.train_y = np.vstack( [self.train_y, y] )
		
		self.class_count = np.max(self.train_y) + 1
	
	def predict_proba(self, X):
		epsilon = 0.00000000001 # threshold for normalization
		
		proba = np.zeros( (X.shape[0], self.class_count) )
		
		distances, neighbor_indices = self.kneighbors(X) # todo distance return optional
		
		for row_index, prob_row in enumerate(proba):
			for n_index in neighbor_indices[row_index, :]:
				n_class = self.train_y[n_index]
				prob_row[n_class] += 1.0
		
		norm = np.linalg.norm(proba, axis=1)
		norm[norm < epsilon] = 1.0
		proba /= norm.reshape( (-1,1) )
		
		return proba
	
	def kneighbors(self, X, n_neighbors=None, return_distance=True):
		if n_neighbors is None:
			n_neighbors = self.n_neighbors
		
		distances = np.empty( (X.shape[0], n_neighbors) )
		indices = np.empty( (X.shape[0], n_neighbors) )
		
		for row_index, point in enumerate(X):
			initial_neighbors, other_neighbors = np.split(self.train_X, [n_neighbors])
			neighbors = [ (self.metric(point, neigh), ind) for ind, neigh in enumerate(initial_neighbors) ]
			neighbors.sort(key=lambda x: x[0])
			
			for neigh_index, neigh in enumerate(other_neighbors):
				dist = self.metric(point, neigh)
				if dist < neighbors[-1][0]:
					true_index = neigh_index + n_neighbors # because other_neighbors is just a part of all neighbors
					neighbors[-1] = (dist, true_index)
					neighbors.sort(key=lambda x: x[0])
			
			d, n_ind = zip(*neighbors)
			distances[row_index, :] = d
			indices[row_index, :] = n_ind
		
		return distances, indices















