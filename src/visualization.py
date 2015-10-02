import matplotlib.pyplot as plt
import numpy as np
import importer as im
import data_processing as dp
from scipy.misc import imread

def create_crime_lists(crimes, id_to_crime_dict):
	'''Creates a dictionary which saves lists of locations of one crime each'''
	crime_dict = {}
	
	for c in crimes:
		try:
			crime_dict[id_to_crime_dict[c[2]]].append((c[0], c[1]))
		except KeyError:
			crime_dict[id_to_crime_dict[c[2]]] = [(c[0], c[1])]
	
	return crime_dict

def plot_crimes(crimes, title):
	'''Plots passed crimes as dots on 2D plane'''
	fig = plt.figure(figsize=(12,9))
	fig.suptitle("Crimes in San Francisco", fontsize=14, fontweight = "bold")
	
	ax = fig.add_subplot(111)
	ax.set_title(title)
	
	ax.set_xlabel('Horizontal GPS coordinates')
	ax.set_ylabel('Vertical GPS coordinates')
	
	plt.scatter(crimes[:,0], crimes[:,1])
	plt.grid(True)
	
	ax.set_xlim([-122.535908,-122.347306])
	ax.set_ylim([37.696850,37.839763])
	
	img = imread("../img/sf.png")
	plt.imshow(img,zorder=0,extent=[-122.535908, -122.347306, 37.696850, 37.839763])
	
	plt.show()

if __name__ == '__main__':
	path = "../data/train.csv"
	data = im.read_labeled(path)
	data = dp.vectorize(data, 1, features=[('latitude', 7), ('longitude', 8)])
	crime_to_id_dict = data.next()
	data = im.to_numpy_array(data)
	id_to_crime_dict = {value: key for key, value in crime_to_id_dict.items()}
	crime_lists = create_crime_lists(data, id_to_crime_dict)

	for cl in crime_lists:
		arr = np.asarray(crime_lists[cl])
		plot_crimes(arr,cl)