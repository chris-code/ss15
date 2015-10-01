import math
import numpy as np
import sklearn as skl
import sklearn.neighbors
import sklearn.cross_validation as cv
import importer
import data_processing as dapo
import evaluation as eval

def distance_in_mod(a, b, m):
	if a > b:
		return min( a - b, m - (a - b) )
	else:
		return min( b - a, m - (b - a) )

# Expects a and b to be of the format
# (latitude, longitude, day_of_month, day_of_week, time_of_day, street_data_1, street_data_1, street_flag)
def distance_function(a, b):
	dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
	dist += distance_in_mod(a[2], b[2], modulo_for_day)
	dist += distance_in_mod(a[3], b[3], modulo_for_day_of_week)
	dist += distance_in_mod(a[4], b[4], modulo_for_time)
	
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

def train(neighbor_counts = [1]):
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
	return knn_c.predict_proba(data)

##### Training phase #####
# In this phase

# Load training data
train_path = 'data/train.csv'
predictions_path = 'data/predictions.csv'
data = importer.read(train_path, 3000)
data = dapo.vectorize(data, 1, features=[('latitude', 7), ('longitude', 8), ('day', 0), ('day_of_week', 0), ('time', 0), ('streets', 6)])
crime_to_id_dict = data.next() # FIXME change 1
data = importer.to_numpy_array(data)
data = dapo.ensure_unit_variance(data, columns_to_normalize=(0, 1, 2, 3, 4))

# Separate labels from data
crime_ids = data[:,-1].astype(int)
locations = data[:,:-1]

# Calculate ranges for the modulo used on circular quantities
modulo_for_day = abs( min(locations[:,2]) - max(locations[:,2]) )
modulo_for_day_of_week = abs( min(locations[:,3]) - max(locations[:,3]) )
modulo_for_time = abs( min(locations[:,4]) - max(locations[:,4]) )

# Split into train and test set
loc_train, loc_test, crime_ids_train, crime_ids_test = cv.train_test_split(locations, crime_ids, test_size=0.33)

# Train and evaluate
# neighbor_counts = [43, 83, 123, 163, 203, 243, 283]
neighbor_counts = [43, 83, 123]
# neighbor_counts = [43]
knn_c, neighbor_count = train(neighbor_counts)
predictions = predict(knn_c, loc_test)
ll = eval.logloss(predictions, crime_ids_test)
print('Log loss: {0}'.format(ll))

##### Prediction phase #####
# In this phase

# Load data to predict
test_path = 'data/test.csv'
predictions_path = 'data/predictions.csv'
data = importer.read_unlabled(test_path, 1001)
data = dapo.vectorize(data, None, features=[('latitude', 4), ('longitude', 5), ('day', 0), ('day_of_week', 0), ('time', 0), ('streets', 3)])
data = importer.to_numpy_array(data)
data = dapo.ensure_unit_variance(data, columns_to_normalize=(0, 1, 2, 3, 4))

# Calculate new modulo ranges for circular quantities (see above). These have to include the data used in training, since the
# NN classifier calculates distances between 
modulo_for_day = abs( min( np.hstack([data[:,2], locations[:,2]]) ) - max( np.hstack([data[:,2], locations[:,2]]) ) )
modulo_for_day_of_week = abs( min( np.hstack([data[:,3], locations[:,3]]) ) - max( np.hstack([data[:,3], locations[:,3]]) ) )
modulo_for_time = abs( min( np.hstack([data[:,4], locations[:,4]]) ) - max( np.hstack([data[:,4], locations[:,4]]) ) )

# Train NN classifier on complete training data and use it to predict crime types of the test set.
knn_c = skl.neighbors.KNeighborsClassifier(n_neighbors=neighbor_count, weights='distance', metric='pyfunc', func=distance_function)
knn_c.fit(locations, crime_ids)
predictions = predict(knn_c, data)
importer.write(predictions_path, predictions, crime_to_id_dict)




