import math
import numpy as np
import sklearn as skl
import sklearn.neighbors
import sklearn.cross_validation as cv
import importer
import data_processing as dapo
import evaluation as eval

def distance_in_mod(a, b, m):
	'''Calculates and returns the distance of two values of a circular quantity.'''
	if a > b:
		return min( a - b, m - (a - b) )
	else:
		return min( b - a, m - (b - a) )

# Expects a and b to be of the format
# (latitude, longitude, day_of_month, day_of_week, time_of_day, street_data_1, street_data_1, street_flag)
def distance_function(a, b):
	'''Determines and returns the (scalar) distance of two data points.
	
	Parameters a and b are expected to be subscriptables containing
	(latitude, longitude, day_of_month, day_of_week, time_of_day, street_data_1, street_data_1, street_flag)
	where the first 5 should be normalized to unit variance. For details on the street data format, see
	vectorization documentation.
	
	For longitude and latitude, manhattan distance is used. The distance in the temporal quantities is calculated
	in a wrap-around fashion. The sum of these constitues the distance. Additionally, the street information can
	provide a 'bonus' to two points a and b. Since crime on the same street is assumed to be more similar than crime
	on different streets, even if the distance is comparable, two points on the same street bet a 'bonus'. Their distance
	is divided by 2 if they lie on the same street, and by 3 if they share both streets (same crossroad)
	'''
	dist = abs(a[0] - b[0]) + abs(a[1] - b[1]) # Manhattan distance
	
	# Circular quantities
	dist += distance_in_mod(a[2], b[2], modulo_for_day)
	dist += distance_in_mod(a[3], b[3], modulo_for_day_of_week)
	dist += distance_in_mod(a[4], b[4], modulo_for_time)
	
	# Bonuses for occurences on the same street
	divisor = 1
	if a[7] < 0 and b[7] < 0:
		if abs(a[5] - b[5]) < 0.1: divisor += 1.0
		if abs(a[5] - b[6]) < 0.1: divisor += 1.0
		if abs(a[6] - b[5]) < 0.1: divisor += 1.0
		if abs(a[6] - b[6]) < 0.1: divisor += 1.0
	elif a[7] < 0 and b[7] > 0:
		if abs(a[5] - b[5]) < 0.1: divisor += 1.0
		elif abs(a[6] - b[5]) < 0.1: divisor += 1.0
	elif a[7] > 0 and b[7] < 0:
		if abs(a[5] - b[5]) < 0.1: divisor += 1.0
		elif abs(a[5] - b[6]) < 0.1: divisor += 1.0
	else:
		if abs(a[5] - b[5]) < 0.1: divisor += 1.0
	dist /= divisor
	
	return math.sqrt(dist)

# TODO pass training data?
def train(neighbor_counts = [1]):
	'''Trains multiple NN classifiers and returns the best one and the number of neighbors it uses.
	
	For each integer in neighbor_counts, a NN classifier is trained on loc_train, crime_ids_train and evaluated
	on loc_test, crime_ids_test. The one with the best accuracy is returned along with its neighbor count. The
	classifiers use  distance proportional weights and distance_function to calculate distances. Take not that
	using a custom distance function is rediculously slow.
	'''
	best_score = 0
	for neighbor_count in neighbor_counts:
		knn_c = skl.neighbors.KNeighborsClassifier(n_neighbors=neighbor_count, weights='distance', metric='pyfunc', func=distance_function)
		# knn_c = skl.neighbors.KNeighborsClassifier(n_neighbors=neighbor_count, weights='distance')
		knn_c.fit(loc_train, crime_ids_train)
		score = knn_c.score(loc_test, crime_ids_test)
		print('Score with {0} neighbors: {1}'.format(neighbor_count, score))
		
		if score > best_score:
			best_score = score
			best_knn_c = knn_c
			best_neighbor_count = neighbor_count
		
	return best_knn_c, best_neighbor_count

def predict(knn_c, data):
	'''Uses knn_c to predict and return the class probabilities for all entries in data
	
	knn_c should be a (trained) sklearn.KNeighborsClassifier, and data a numpy array of shape
	(#samples, dimensionalty_of_data). It will return a numpy array of shape (#samples, #crime types),
	which for each sample, indicates the estimated proabilities of of the various learned crime types.
	'''
	return knn_c.predict_proba(data)

if __name__ == '__main__':
	########## Training phase ##########
	# In this phase the original (labeled) training data is split into a new set of training and test data.
	# These sets are used to determine the optimal parameters (number of neighbors) for a NN classifier.
	# For each tested classifier, the score is calculated and printed. Additionally, the log loss as applied
	# by kaggle is calculated and printed for the best classifier.

	train_path = 'data/train.csv'
	predictions_path = 'data/predictions.csv'

	# Load training data
	data = importer.read_labeled(train_path, 50000) # Read at most 50000 data points
	data = dapo.vectorize(data, 1, features=[('latitude', 7), ('longitude', 8), ('day', 0), ('day_of_week', 0), ('time', 0), ('streets', 6)])
	crime_to_id_dict = data.next()
	data = importer.to_numpy_array(data) # Collect data in array
	data = dapo.ensure_unit_variance(data, columns_to_normalize=(0, 1, 2, 3, 4)) # Ensure unit variance in appropriate columns

	# Separate labels from data
	crime_ids = data[:,-1].astype(int) # Crime ids are in the last column, and are integer values
	locations = data[:,:-1] # The rest is data

	# Calculate ranges for the modulo used on circular quantities
	modulo_for_day = abs( min(locations[:,2]) - max(locations[:,2]) )
	modulo_for_day_of_week = abs( min(locations[:,3]) - max(locations[:,3]) )
	modulo_for_time = abs( min(locations[:,4]) - max(locations[:,4]) )

	# Split into train and test set
	loc_train, loc_test, crime_ids_train, crime_ids_test = cv.train_test_split(locations, crime_ids, test_size=0.33)

	# Train and evaluate
	# neighbor_counts = [43, 83, 123, 163, 203, 243, 283]
	neighbor_counts = [43, 83, 123, 163]
	knn_c, neighbor_count = train(neighbor_counts)
	predictions = predict(knn_c, loc_test)
	ll = eval.logloss(predictions, crime_ids_test) # Log loss is the measure applied by kaggle
	print('Log loss: {0}'.format(ll))

	########## Prediction phase #########
	# In this phase, a new NN classifier is trained on the original (complete) data set, with the optimal number of neighbors as determined
	# during the training phase. That is then used to predict the crime types on the orignal test set, the result of which is written to disk.

	test_path = 'data/test.csv'
	predictions_path = 'data/predictions.csv'

	# Load data to predict
	data = importer.read_unlabeled(test_path, 10000) # Read at most 10000 data points to predict crimes on
	data = dapo.vectorize(data, None, features=[('latitude', 4), ('longitude', 5), ('day', 0), ('day_of_week', 0), ('time', 0), ('streets', 3)])
	data = importer.to_numpy_array(data) # Collect data in numpy array
	data = dapo.ensure_unit_variance(data, columns_to_normalize=(0, 1, 2, 3, 4)) # Ensure unit variance in appropriate columns

	# Calculate new modulo ranges for circular quantities (see above). In calculating these ranges, we have to include the data used in training,
	# since the NN classifier calculates distances between the points to be predicted and points used in training.
	modulo_for_day = abs( min( np.hstack([data[:,2], locations[:,2]]) ) - max( np.hstack([data[:,2], locations[:,2]]) ) )
	modulo_for_day_of_week = abs( min( np.hstack([data[:,3], locations[:,3]]) ) - max( np.hstack([data[:,3], locations[:,3]]) ) )
	modulo_for_time = abs( min( np.hstack([data[:,4], locations[:,4]]) ) - max( np.hstack([data[:,4], locations[:,4]]) ) )

	# Train NN classifier on complete training data (with the best number of neighbors) and use it to predict crime types of the test set.
	knn_c = skl.neighbors.KNeighborsClassifier(n_neighbors=neighbor_count, weights='distance', metric='pyfunc', func=distance_function)
	knn_c.fit(locations, crime_ids)
	predictions = predict(knn_c, data)
	importer.write(predictions_path, predictions, crime_to_id_dict) # Write predicted data to disk in format specified by kaggle




