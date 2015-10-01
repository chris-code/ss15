import time
import re
import csv
import numpy as np
import sklearn as skl
import sklearn.preprocessing

def read(path, limit=None):
	with open(path, 'r') as file: # FIXME change 2
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

def vectorize(data, features=['time', 'day', 'month', 'year', 'day_of_week', 'latitude', 'longitude']):
	crime_type_ids = {}
	crime_type_counter = 0
	street_ids = {}
	street_counter = 0
	yield crime_type_ids

	street_type_1 = re.compile(r'(.+) / (.+)')
	street_type_2 = re.compile(r'(.+) Block of (.+)')
	for data_point in data:
		try:
			crime_type_id = crime_type_ids[data_point[1]]
		except KeyError:
			crime_type_ids[data_point[1]] = crime_type_counter
			crime_type_counter += 1
			crime_type_id = crime_type_ids[data_point[1]]
		
		vec = [crime_type_id]
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
			elif feature == 'latitude':
				vec.append(data_point[7])
			elif feature == 'longitude':
				vec.append(data_point[8])
			elif feature == 'streets':
				type1_match = street_type_1.match(data_point[6])
				if type1_match is not None:
					street1, street2 = type1_match.group(1, 2)
					if street1 not in street_ids:
						street_ids[street1] = street_counter
						street_counter += 1
					if street2 not in street_ids:
						street_ids[street2] = street_counter
						street_counter += 1
					s1_id = street_ids[street1]
					s2_id = street_ids[street2]
					vec.append(s1_id)
					vec.append(s2_id)
					vec.append(-1)
				else:
					type2_match = street_type_2.match(data_point[6])
					if type2_match is not None:
						block, street = type2_match.group(1, 2)
						block = int(block)
						if street not in street_ids:
							street_ids[street] = street_counter
							street_counter += 1
						s_id = street_ids[street]
						vec.append(s_id)
						vec.append(block)
						vec.append(1)
					else:
						raise 'Unknown street format: {0}'.format(data_point[6])
			else:
				raise 'Feature not supported!'
		yield vec

def preprocess(data, lat, long):
	# define outermost coordinates
	SOUTH = {'y': 37.696850, 'x': -122.440464}
	EAST = {'y': 37.764893, 'x': -122.347306} 
	NORTH = {'y': 37.839763, 'x': -122.424554}
	WEST = {'y': 37.728356, 'x': -122.535908}
	
	for data_point in data:
		if data_point[lat] < WEST['x'] \
		or data_point[lat] > EAST['x'] \
		or data_point[long] < SOUTH['y'] \
		or data_point[long] > NORTH['y']:
			continue
		yield data_point
	
def to_numpy_array(data):
	collected_data = [data_point for data_point in data]
	return np.asarray(collected_data)

def ensure_unit_variance(data, columns_to_normalize):
	scaled_data = skl.preprocessing.scale(data)
	
	new_data = data.copy()
	for column in columns_to_normalize:
		new_data[:,column] = scaled_data[:,column]
	return new_data
	
	# return np.hstack( [data[:,0].reshape(-1,1), data_scaled] )

def write(path, predictions, crime_to_id):
	# id_to_crime_dict = {value: key for key, value in crime_to_id_dict}
	first_line = [crime for crime in sorted(crime_to_id, key=crime_to_id.get)]
	first_line.insert(0, 'Id')

	with open(path, 'w') as file: # FIXME change 3
		csv_writer = csv.writer(file, delimiter=',')
		
		csv_writer.writerow(first_line)
		for index, probabilities in enumerate(predictions):
			row = probabilities.tolist()
			row.insert(0, index)
			csv_writer.writerow( row )
	
# Show lower and upper limit of raw vs. normalized time of day
# data = to_numpy_array(vectorize(read('data/train.csv', 10000)))
# print('Min: {0} Max: {1}'.format(min(data[:,1]), max(data[:,1])))
# rescaled_data = ensure_unit_variance(data)
# print('Min: {0} Max: {1}'.format(min(rescaled_data[:,1]), max(rescaled_data[:,1])))

# Plot histogram of raw vs. rescaled time of day
# import matplotlib.pyplot as plt
# fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
# ax0.hist(data[:,1], 20)
# ax0.set_title('Raw')
# ax1.hist(rescaled_data[:,1], 20)
# ax1.set_title('Rescaled')
# plt.show()

# Print counts for different crimes
# crime_counter = {}
# for data_point in read('data/train.csv', 10000):
	# try:
		# crime_counter[data_point[1]] += 1
	# except KeyError:
		# crime_counter[data_point[1]] = 1

# for key, value in crime_counter.items():
	# print('{0}: {1}'.format(key, value))
