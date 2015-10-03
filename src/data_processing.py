import re
import sklearn as skl
import sklearn.preprocessing

def vectorize(data, label_column, features):
	'''Generator function that extracts and returns a selectible set of features for each data point in the data parameter.
	
	Returns a vector containing values for the selected features. The order in the features parameter is indicative of the order in
	the resulting vector (except when using the 'streets' feature, see below).
	
	data
	data is a sequence of subscriptables. Each subscriptable usually represents a line read from a .csv file.
	
	label_column
	label_column is None if the data is unlabeled. Otherwise, it is the index under which the label is found in each subscriptable.
	The label is assumed to be a string and will be mapped to an integer unique to each unique string. This integer is appended as the
	last element in the output vector.
	There is one more thing about this parameter. If it is not none, the first element yielded by this generator function is a dictionary
	that maps crime type strings to unambiguous ids. It is empty until more elements are extracted from this generator.
	
	features
	features is an iterable of 2-tuples (feature, index). It indicates
	- which features to extract (first part of each tuple)
	- under which index information needed for that feature is found (second part of each tuple)
	- in which order to place the features in the resulting feature vector (given by the order in the iterable)
	
	Available features are: 'time', 'day', 'month', 'year', 'day_of_week', 'latitude', 'longitude', 'streets'.
	The features 'time', 'day', 'month', 'year' and 'day_of_week' are extracted from a time.struct_time object, so their associated
	index in the (feature, index) tuple is usually identical. 'latitude' and 'longitude' are expected to be floats and are used 'as is'.
	
	The 'streets' feature is a bit special in that it doesn't produce a single value in the output vector, but three. There are two types
	of street designation formats in the data set:
	- STREET_1 / STREET_2
	- Xth block of STREET
	In the former case, two unique ids for the streets are appended to the output vector, followed by a -1. In the latter case, the street
	id and then the block number is appended to the vector, followed by a +1.
	'''
	
	street_type_1 = re.compile(r'(.+) / (.+)') # Regular expression to recognize street designations of the form 'STREET_1 / STREET_2'
	street_type_2 = re.compile(r'(.+) Block of (.+)') # as above, for ' Xth block of STREET'
	crime_type_ids = {} # Dictionary unambiguously mapping crime type strings to integer ids
	crime_type_counter = 0 # Counts how many different types of crime have been found.
	street_ids = {} # Dictionary unambiguously mapping street name strings to integer ids
	street_counter = 0 # Counts how many unique street names have been found.
	
	# Provide caller with the dictionary if appropriate.
	#~ if label_column is not None:
		#~ yield crime_type_ids

	vectorized_data = []
	vectorized_labels = []
	for data_point in data:
		# Create vector and append all requested features
		vec = []
		for feature, column in features:
			if feature == 'time':
				time = data_point[column].tm_hour * 60 + data_point[column].tm_min # Time in minutes since 00:00
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
			elif feature == 'latitude':
				vec.append(data_point[column])
			elif feature == 'longitude':
				vec.append(data_point[column])
			elif feature == 'streets':
				type1_match = street_type_1.match(data_point[column])
				if type1_match is not None: # Street designation is of the form 'STREET_1 / STREET_2'
					street1, street2 = type1_match.group(1, 2) # fetch components
					if street1 not in street_ids: # Get / create street id
						street_ids[street1] = street_counter
						street_counter += 1
					if street2 not in street_ids: # Get / create street id
						street_ids[street2] = street_counter
						street_counter += 1
					s1_id = street_ids[street1]
					s2_id = street_ids[street2]
					vec.append(s1_id)
					vec.append(s2_id)
					vec.append(-1)
				else: # Street designation is of the form 'Xth block of STREET'
					type2_match = street_type_2.match(data_point[column])
					if type2_match is not None:
						block, street = type2_match.group(1, 2) # fetch components
						block = int(block)
						if street not in street_ids: # Get / create street id
							street_ids[street] = street_counter
							street_counter += 1
						s_id = street_ids[street]
						vec.append(s_id)
						vec.append(block)
						vec.append(1)
					else: # Street designation is in neither format
						raise 'Unknown street format: {0}'.format(data_point[6])
			else: # Caller has requested an unknown feature
				raise 'Feature not supported!'
		vectorized_data.append(vec)
		
		# Get crime id from dictionary, or make new one if neccessary.
		if label_column is not None:
			try:
				crime_type_id = crime_type_ids[data_point[label_column]]
			except KeyError:
				crime_type_ids[data_point[label_column]] = crime_type_counter
				crime_type_counter += 1
				crime_type_id = crime_type_ids[data_point[label_column]]
			vectorized_labels.append(crime_type_id)
		
	if label_column is not None:
		return crime_type_ids, vectorized_labels, vectorized_data
	else:
		return vectorized_data

def remove_outliers(data, lat_index, long_index):
	'''Generator function that yields every item in the sequence data, if it is within the specified coordinates.'''
	# define outermost coordinates
	SOUTH = {'y': 37.696850, 'x': -122.440464}
	EAST = {'y': 37.764893, 'x': -122.347306} 
	NORTH = {'y': 37.839763, 'x': -122.424554}
	WEST = {'y': 37.728356, 'x': -122.535908}
	
	for data_point in data:
		if data_point[lat_index] < WEST['x'] \
		or data_point[lat_index] > EAST['x'] \
		or data_point[long_index] < SOUTH['y'] \
		or data_point[long_index] > NORTH['y']:
			continue # data point is out of bounds, skip it
		
		yield data_point

def ensure_unit_variance(data, columns_to_normalize):
	'''Returns a version of data where all indicated columns are made to be mean-free and have unit variance.
	
	data is a numpy array of shape (#samples, #features)
	columns_to_normalize is an iterable of column indices
	'''
	scaled_data = skl.preprocessing.scale(data)
	
	new_data = data.copy()
	for column in columns_to_normalize:
		new_data[:,column] = scaled_data[:,column]
	return new_data











