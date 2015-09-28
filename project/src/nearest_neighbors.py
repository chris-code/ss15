import numpy as np
import sklearn as skl
import sklearn.neighbors
import sklearn.cross_validation as cv
import importer

path = 'data/train.csv'

locations = []
crime_ids = []
crime_id_map = {}
id_counter = 0
#for data_point in importer.read(path, 10000):
for data_point in importer.read(path):
	try:
		crime_id = crime_id_map[data_point[1]]
	except KeyError:
		crime_id_map[data_point[1]] = id_counter
		crime_id = crime_id_map[data_point[1]]
		id_counter += 1
	
	crime_ids.append(crime_id)
	locations.append( (data_point[7], data_point[8]) )
locations = np.asarray(locations)
crime_ids = np.asarray(crime_ids)

loc_train, loc_test, crime_ids_train, crime_ids_test = cv.train_test_split(locations, crime_ids, test_size=0.33)

#def distance_function():	
#distance_metric = skl.neighbors.DistanceMetric.get_metric('pyfunc', distance_function)

neighbor_counts = [3, 43, 83, 123, 163, 203]
for neighbor_count in neighbor_counts:
	#knnc = skl.neighbors.KNeighborsClassifier(n_neighbors=neighbor_count, metric=)
	knnc = skl.neighbors.KNeighborsClassifier(n_neighbors=neighbor_count)
	knnc.fit(loc_train, crime_ids_train)
	score = knnc.score(loc_test, crime_ids_test)
	print('Score with {0} neighbors: {1}'.format(neighbor_count, score))

#knnc = skl.neighbors.KNeighborsClassifier()
#knnc.fit(loc_train, crime_ids_train)
#score = knnc.score(loc_test, crime_ids_test)
#print('Score: ', score)