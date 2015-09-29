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

def vectorize(data, features=[]):
	crime_type_ids = {}
	crime_type_counter = 0
	
	#day_of_week_ids = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
	
	for data_point in data:
		try:
			crime_type_id = crime_type_ids[data_point[1]]
		except KeyError:
			crime_type_ids[data_point[1]] = crime_type_counter
			crime_type_counter += 1
			crime_type_id = crime_type_ids[data_point[1]]
		
		vec = []
		for feature in features:
			if feature == 'time':
				time = data_point[0].tm_hour * 60 + data_point[0].tm_min
				vec.append(time)
			elif feature == 'day':
				day = data_point[0].tm_mday
				vec.append(day)
			elif feature == 'month':
				month = data_point[0].tm_mon
				vec.append(month)
			elif feature == 'year':
				year = data_point[0].tm_year
				vec.append(year)
			elif feature == 'day_of_week':
				vec.append(data_point[0].tm_wday)
			elif feature == 'crime_type_id':
				vec.append(crime_type_id)
			elif feature == 'latitude':
				vec.append(data_point[7])
			elif feature == 'longitude':
				vec.append(data_point[8])
			else:
				raise 'Feature not supported!'
		yield vec

def to_numpy_array(data):
	collected_data = [data_point for data_point in data]
	return np.asarray(collected_data)

# crime_counter = {}
# for data_point in read('data/train.csv', 10000):
	# try:
		# crime_counter[data_point[1]] += 1
	# except KeyError:
		# crime_counter[data_point[1]] = 1

# for key, value in crime_counter.items():
	# print('{0}: {1}'.format(key, value))