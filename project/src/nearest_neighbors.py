import math
import numpy as np
import sklearn as skl
import sklearn.neighbors
import sklearn.cross_validation as cv
import importer

def distance_in_mod(a, b, m):
	if a > b:
		return min( a - b, m - (a - b) )
	else:
		return min( b - a, m - (b - a) )

# Expects a and b to be of the format
# (latitude, longitude, day_of_month, day_of_week, time_of_day)
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
			best_knn_c = knn_c
		
	return best_knn_c

def predict(knn_c, data):
	return knn_c.predict_proba(data)

def logloss(predictions, truth):
	ll = 0.0
	for i in range(predictions.shape[0]):
		true_crime_id = truth[i]
		try:
			prob = predictions[i, true_crime_id]
		except IndexError:
			prob = 0
		prob = max( min(prob, 1 - 10**(-15)) , 10**(-15))
		ll += math.log(prob)
	return (-1.0) * ll / predictions.shape[0]

# Load data
train_path = 'data/train.csv'
predictions_path = 'data/predictions.csv'
data = importer.read(train_path, 3000)
data = importer.vectorize(data, features=['latitude', 'longitude', 'day', 'day_of_week', 'time', 'streets'])
crime_to_id_dict = data.__next__() # FIXME change 1
data = importer.to_numpy_array(data)
data = importer.ensure_unit_variance(data, columns_to_normalize=(1, 2, 3, 4, 5))

crime_ids = data[:,0].astype(int)
locations = data[:,1:]
modulo_for_day = abs(min(locations[:,2]) - max(locations[:,2]))
modulo_for_day_of_week = abs(min(locations[:,3]) - max(locations[:,3]))
modulo_for_time = abs(min(locations[:,4]) - max(locations[:,4]))

# Split into train and test set
loc_train, loc_test, crime_ids_train, crime_ids_test = cv.train_test_split(locations, crime_ids, test_size=0.33)

# Train and evaluate
# neighbor_counts = [43, 83, 123, 163, 203, 243, 283]
neighbor_counts = [43, 83, 123]
knn_c = train(neighbor_counts)
predictions = predict(knn_c, loc_test)
ll = logloss(predictions, crime_ids_test)
print('Log loss: {0}'.format(ll))
importer.write(predictions_path, predictions, crime_to_id_dict)












