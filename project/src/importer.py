import time
import csv
import numpy as np

def read(path, limit=None):
	with open(path) as file:
		descriptions = file.readline().split(',')
		csv_reader = csv.reader(file)
		for index, data_point in enumerate(csv_reader):
			if limit is not None and index >= limit:
				break
			
			date = time.strptime(data_point[0], '%Y-%m-%d %H:%M:%S')
			category = data_point[1]
			descript = data_point[2]
			day_of_week = data_point[3]
			pd_district = data_point[4]
			resolution = data_point[5]
			adress = data_point[6]
			X = float(data_point[7])
			Y = float(data_point[8])
			yield (date, category, descript, day_of_week, pd_district, resolution, adress, X, Y)

def vectorize(data):
	crime_type_ids = {}
	crime_type_counter = 0
	
	for data_point in data:
		try:
			crime_type_id = crime_type_ids[data_point[1]]
		except KeyError:
			crime_type_ids[data_point[1]] = crime_type_counter
			crime_type_counter += 1
			crime_type_id = crime_type_ids[data_point[1]]
		
		time = data_point[0].tm_hour * 60**2 + data_point[0].tm_min * 60 + data_point[0].tm_sec
		
		yield (time, crime_type_id, data_point[7], data_point[8])

def preprocess(data):
	# define outermost coordinates
	SOUTH = {'y': 37.696850, 'x': -122.440464}
	EAST = {'y': 37.764893, 'x': -122.347306} 
	NORTH = {'y': 37.839763, 'x': -122.424554}
	WEST = {'y': 37.728356, 'x': -122.535908}
	
	for data_point in data:
		if data_point[2] < WEST['x'] \
		or data_point[2] > EAST['x'] \
		or data_point[3] < SOUTH['y'] \
		or data_point[3] > NORTH['y']:
			continue
		yield data_point
			
def to_numpy_array(data):
	collected_data = [data_point for data_point in data]
	return np.asarray(collected_data)

# data = to_numpy_array(preprocess(vectorize(read('data/train.csv'))))
# print(data.shape)