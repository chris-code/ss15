import time
import csv
import numpy as np

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
	
def to_numpy_array(data):
	collected_data = [data_point for data_point in data]
	return np.asarray(collected_data)

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
