import time
import csv

def read(path):
	with open(path) as file:
		descriptions = file.readline().split(',')
		csv_reader = csv.reader(file)
		for data_point in csv_reader:
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