import math
import numpy as np
#~ import sklearn as skl
#~ import sklearn.neighbors
import sklearn.cross_validation as cv
import importer
import data_processing as dapo
import evaluation as ev
import nearest_neighbors_c as nnc

def calc_mod_ranges(data, columns=None):
	if columns is None:
		columns = range(data.shape[1])
	
	ranges = []
	for c in columns:
		value = abs( min(data[:,c]) - max(data[:,c]) )
		ranges.append(value)
	return ranges

def distance_in_mod(a, b, m):
	'''Calculates and returns the distance of two values of a circular quantity.'''
	if a > b:
		return min( a - b, m - (a - b) )
	else:
		return min( b - a, m - (b - a) )

def distance_function(a, b, modulae, weights=(1.0, 1.0, 1.0, 1.0)):
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
	location_weight, day_weight, day_of_week_weight, time_weight = weights
	
	# Spatial distance
	dist = location_weight * abs(a[0] - b[0]) + abs(a[1] - b[1]) # Manhattan distance
	
	# Circular quantities
	modulo_for_day, modulo_for_day_of_week, modulo_for_time = modulae
	dist += day_weight * distance_in_mod(a[2], b[2], modulo_for_day)
	dist += day_of_week_weight * distance_in_mod(a[3], b[3], modulo_for_day_of_week)
	dist += time_weight * distance_in_mod(a[4], b[4], modulo_for_time)
	
	# Bonuses for occurences on the same street
	divisor = 1.0
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

def grid_search(data, labels, modulae, neighbor_counts, weight_lists):
	'''Trains multiple NN classifiers and returns the best one and the number of neighbors it uses.
	
	For each integer in neighbor_counts, a NN classifier is trained on loc_train, crime_ids_train and evaluated
	on loc_test, crime_ids_test. The one with the best accuracy is returned along with its neighbor count. The
	classifiers use  distance proportional weights and distance_function to calculate distances. Take not that
	using a custom distance function is rediculously slow.
	'''
	location_weights, day_weights, day_of_week_weights, time_weights = weight_lists
	
	# Split into train and test set
	loc_train, loc_test, crime_ids_train, crime_ids_test = cv.train_test_split(data, labels, test_size=0.33)
	
	best_log_loss = 10**10
	for nc in neighbor_counts:
		for lw in location_weights:
			for dw in day_weights:
				for doww in day_of_week_weights:
					for tw in time_weights:
						weights = (lw, dw, doww, tw)
						met_parms = {'modulae': modulae, 'weights': weights}
						#~ knn_c = skl.neighbors.KNeighborsClassifier(n_neighbors=nc, weights='distance', metric='pyfunc', func=distance_function, metric_params=met_parms)
						#~ knn_c = skl.neighbors.KNeighborsClassifier(n_neighbors=nc, metric='pyfunc', func=distance_function, metric_params=met_parms)
						knn_c = nnc.Nearest_Neighbor_Classifier(n_neighbors=nc, metric=distance_function, metric_params=met_parms)
						knn_c.fit(loc_train, crime_ids_train)
						predictions = knn_c.predict_proba(loc_test)
						log_loss = ev.logloss(predictions, crime_ids_test)
						
						if log_loss < best_log_loss:
							best_log_loss = log_loss
							best_neighbor_count = nc
							best_weights = lw, dw, doww, tw
		
	return best_log_loss, best_neighbor_count, best_weights

def read_training_data(data_path, data_limit):
	# Load training data
	data_train = importer.read_labeled(data_path, data_limit) # Read at most 3000 data points
	features=[('latitude', 7), ('longitude', 8), ('day', 0), ('day_of_week', 0), ('time', 0), ('streets', 6)]
	crime_to_id, labels, data_train = dapo.vectorize(data_train, label_column=1, features=features)
	data_train = importer.to_numpy_array(data_train) # Collect data in array
	labels = importer.to_numpy_array(labels).astype(int)
	data_train = dapo.ensure_unit_variance(data_train, columns_to_normalize=(0, 1, 2, 3, 4)) # Ensure unit variance in appropriate columns

	# Calculate ranges for the modulo used on circular quantities
	modulae = calc_mod_ranges(data_train, (2, 3, 4))

	return data_train, labels, modulae, crime_to_id

def read_test_data(data_path, data_limit, data_train):
	# Load data to predict
	data_test = importer.read_unlabeled(test_path, data_limit) # Read at most 1000 data points to predict crimes on
	features = [('latitude', 4), ('longitude', 5), ('day', 0), ('day_of_week', 0), ('time', 0), ('streets', 3)]
	data_test = dapo.vectorize(data_test, label_column=None, features=features)
	data_test = importer.to_numpy_array(data_test) # Collect data in numpy array
	data_test = dapo.ensure_unit_variance(data_test, columns_to_normalize=(0, 1, 2, 3, 4)) # Ensure unit variance in appropriate columns

	# Calculate new modulo ranges for circular quantities (see above). In calculating these ranges, we have to include the data used in training,
	# since the NN classifier calculates distances between the points to be predicted and points used in training.
	modulae = calc_mod_ranges(np.vstack( [data_test, data_train] ), (2, 3, 4))
	
	return data_test, modulae

if __name__ == '__main__':
	# Optimize parameters
	train_path = '../data/train.csv'
	data_train, labels, modulae, crime_to_id = read_training_data(train_path, data_limit=1000)
	neighbor_counts = [163]
	location_weights, day_weights, day_of_week_weights, time_weights = [0.0325], [0.055], [0.0775], [1.0]
	weight_lists = location_weights, day_weights, day_of_week_weights, time_weights
	log_loss, neighbor_count, weights = grid_search(data_train, labels, modulae, neighbor_counts, weight_lists)
	print('Best log loss of {0} achieved with'.format(log_loss))
	print('{0} neighbors and weights {1}'.format(neighbor_count, weights))

	#~ # Train knn with those parameters
	#~ test_path = '../data/test.csv'
	#~ data_test, modulae = read_test_data(test_path, data_limit=1000, data_train=data_train)
	#~ knn_c = skl.neighbors.KNeighborsClassifier(n_neighbors=neighbor_count, weights='distance', metric='pyfunc', func=distance_function, metric_params={'modulae': modulae})
	#~ knn_c.fit(data_train, labels)
	#~ 
	#~ # Use it to predict on data
	#~ predictions_path = '../data/predictions.csv'
	#~ predictions = knn_c.predict_proba(data_test)
	#~ importer.write(predictions_path, predictions, crime_to_id)




