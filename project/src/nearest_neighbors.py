import math
import numpy as np
import sklearn as skl
import sklearn.neighbors
import sklearn.cross_validation as cv
import importer

path = 'data/train.csv'
data = importer.to_numpy_array(importer.vectorize(importer.read(path, 10000), features=['crime_type_id', 'time', 'latitude', 'longitude', 'day_of_week'] ))
# data = importer.to_numpy_array(importer.vectorize(importer.read(path), features=['crime_type_id', 'time', 'latitude', 'longitude', 'day_of_week'] ))
crime_ids = data[:,0]
locations = data[:,1:]

loc_train, loc_test, crime_ids_train, crime_ids_test = cv.train_test_split(locations, crime_ids, test_size=0.33)

def distance_function(a, b):
	dist = (a[1] - b[1])**2 + (a[2] - b[2])**2
	max_t = max(a[0], b[0])
	min_t = min(a[0], b[0])
	dist += min( (0.001 * (max_t - min_t))**2, ( (0.001 * (min_t + 24 * 60**2 - max_t) ))**2 )
	return math.sqrt(dist)

neighbor_counts = [3, 43, 83, 123, 163, 203, 243, 283]
for neighbor_count in neighbor_counts:
	knnc = skl.neighbors.KNeighborsClassifier(n_neighbors=neighbor_count, weights='distance', metric='pyfunc', func=distance_function)
	knnc.fit(loc_train, crime_ids_train)
	score = knnc.score(loc_test, crime_ids_test)
	print('Score with {0} neighbors: {1}'.format(neighbor_count, score))