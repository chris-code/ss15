import re
import sklearn as skl
import sklearn.preprocessing

def vectorize(data, label_column, features=[('time', 0), ('day', 0), ('month', 0), ('year', 0), ('day_of_week', 0), ('latitude', 7), ('longitude', 8)]):
	crime_type_ids = {}
	crime_type_counter = 0
	street_ids = {}
	street_counter = 0
	
	if label_column is not None:
		yield crime_type_ids

	street_type_1 = re.compile(r'(.+) / (.+)')
	street_type_2 = re.compile(r'(.+) Block of (.+)')
	for data_point in data:
		if label_column is not None:
			try:
				crime_type_id = crime_type_ids[data_point[label_column]]
			except KeyError:
				crime_type_ids[data_point[label_column]] = crime_type_counter
				crime_type_counter += 1
				crime_type_id = crime_type_ids[data_point[label_column]]
			
			vec = [crime_type_id]
		else:
			vec = []
		for feature, column in features:
			if feature == 'time':
				time = data_point[column].tm_hour * 60 + data_point[column].tm_min
				vec.append(time)
			elif feature == 'day':
				day = data_point[column].tm_mday
				vec.append(day)
			elif feature == 'month':
				month = data_point[column].tm_mon
				vec.append(month)
			elif feature == 'year':
				year = data_point[column].tm_year
				vec.append(year)
			elif feature == 'day_of_week':
				vec.append(data_point[column].tm_wday)
				# vec.append(data_point[column])
			elif feature == 'latitude':
				vec.append(data_point[column])
			elif feature == 'longitude':
				vec.append(data_point[column])
			elif feature == 'streets':
				type1_match = street_type_1.match(data_point[column])
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
					type2_match = street_type_2.match(data_point[column])
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

def remove_outliers(data, lat, long):
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

def ensure_unit_variance(data, columns_to_normalize):
	scaled_data = skl.preprocessing.scale(data)
	
	new_data = data.copy()
	for column in columns_to_normalize:
		new_data[:,column] = scaled_data[:,column]
	return new_data