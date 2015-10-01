import time
import csv
import numpy as np

def read_labled(path, limit=None):
	'''Returns a sequence of tuples, each one representing one data point in the .csv file at path.
	
	This is intended for reading the training data . Each tuple is of the form
	(date, category, descript, day_of_week, pd_district, resolution, adress, latitude, longitude)
	where date is a time.struct_time object, category through adress are strings, and latitude and longitude are floats.
	The number of data points read can be limited to a specific number with the limit parameter.
	'''
	with open(path, 'r') as file:
		descriptions = file.readline().split(',') # The first line of the .csv file. We discard this.
		csv_reader = csv.reader(file)
		
		for index, data_point in enumerate(csv_reader):
			if limit is not None and index >= limit:
				break # Yield no more data points
			
			date = time.strptime(data_point[0], '%Y-%m-%d %H:%M:%S') # Convert date string to time.struct_time
			category = data_point[1]
			descript = data_point[2]
			day_of_week = data_point[3]
			pd_district = data_point[4]
			resolution = data_point[5]
			adress = data_point[6]
			latitude = float(data_point[7])
			longitude = float(data_point[8])
			yield (date, category, descript, day_of_week, pd_district, resolution, adress, latitude, longitude)

def read_unlabled(path, limit=None):
	'''Returns a sequence of tuples, each one representing one data point in the .csv file at path.
	
	This is intended for reading the test data. Each tuple is of the form
	(date, day_of_week, pd_district, adress, latitude, longitude)
	where date is a time.struct_time object, day_of_week, pd_district and adress are strings, and
	latitude and longitude are floats.
	The number of data points read can be limited to a specific number with the limit parameter.
	'''
	with open(path, 'r') as file:
		descriptions = file.readline().split(',')
		csv_reader = csv.reader(file)
		
		for index, data_point in enumerate(csv_reader):
			if limit is not None and index >= limit:
				break # Yield no more data points
			
			date = time.strptime(data_point[1], '%Y-%m-%d %H:%M:%S') # Convert date string to time.struct_time
			day_of_week = data_point[2]
			pd_district = data_point[3]
			adress = data_point[4]
			latitude = float(data_point[5])
			longitude = float(data_point[6])
			yield (date, day_of_week, pd_district, adress, latitude, longitude)
			
def to_numpy_array(data):
	'''Generates and returns a single numpy array from the sequence of vectors in data.'''
	collected_data = [data_point for data_point in data]
	return np.asarray(collected_data)

def write(path, predictions, crime_to_id):
	'''Writes the predictions to a .csv file at path, with order given by crime_to_id.
	
	path indicates a storage location on disk.
	predictions is expected to be a numpy array containing probability predictions for each sample as its rows.
	crime_to_id is expected to behave like a dictionary with the crime strings as keys and the ids as values. It
	should map a crime string to the column in the predictions parameter where its probability is stored, e.g.
	crime_to_id['THEFT / LARCENY'] == 4 if the probabilities for theft and larceny are stored in probabilites[:,6].
	
	From this, this function constructs a file on disk that contains the appropriate header, ids and probabilities
	in the format required by kaggle.
	'''
	
	# Liste of crime types ordered by their id. This is because predictions has its columns sorted by id as well.
	first_line = [crime for crime in sorted(crime_to_id, key=crime_to_id.get)]
	first_line.insert(0, 'Id')

	with open(path, 'w') as file: # FIXME change 3
		csv_writer = csv.writer(file, delimiter=',')
		
		csv_writer.writerow(first_line)
		for index, probabilities in enumerate(predictions):
			row = probabilities.tolist()
			row.insert(0, index)
			csv_writer.writerow( row )















