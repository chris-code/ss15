import matplotlib.pyplot as plt
import numpy as np
import importer as im

def create_crime_lists(it):
	crime_dict = {}
	
	for c in it:
		try:
			crime_dict[c[1]].append((c[7], c[8]))
		except KeyError:
			crime_dict[c[1]] = [(c[7], c[8])]
	
	return crime_dict


def plot_crimes(crimes, title):
	'''Plots passed crimes as dots on 2d plane'''
	# define outermost coordinates
	#SOUTH = {'y': 37.696850, 'x': -122.440464}
	#EAST = {'y': 37.728084, 'x': -122.346050}
	#NORTH = {'y': 37.820351, 'x': -122.446644}
	#WEST = {'y': 37.728356, 'x': -122.535908}
	
	fig = plt.figure()
	fig.suptitle('Crimes in San Francisco', fontsize=14, fontweight='bold')

	ax = fig.add_subplot(111)
	ax.set_title(title)
	
	ax.set_xlabel('Horizontal GPS coordinates')
	ax.set_ylabel('Vertical GPS coordinates')
	
	plt.scatter(crimes[:,0], crimes[:,1])
	plt.grid(True)
	
	plt.show()
	
	# TODO: SF Map
	
crime_lists = create_crime_lists(im.read('../data/train.csv'))
for cl in crime_lists:
	arr = np.asarray(crime_lists[cl])
	plot_crimes(arr, cl)