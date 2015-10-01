import matplotlib.pyplot as plt
import numpy as np
import importer as im

def create_crime_lists(it, itcd):
	crime_dict = {}
	
	for c in it:
		try:
			crime_dict[itcd[c[0]]].append((c[1], c[2]))
		except KeyError:
			crime_dict[itcd[c[0]]] = [(c[1], c[2])]
	
	return crime_dict


def plot_crimes(crimes, title):
	'''Plots passed crimes as dots on 2d plane'''
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

path = "../data/train.csv"
data = im.vectorize(im.read(path, 30000), ['latitude', 'longitude', 'time', 'day', 'month', 'year', 'day_of_week'])
crime_to_id_dict = data.next()
id_to_crime_dict = [(v,k) for (k,v) in crime_to_id_dict.items()]
data = im.to_numpy_array(im.preprocess(data, 1, 2))
data = im.ensure_unit_variance(data)

print crime_to_id_dict
print id_to_crime_dict

Y = data[:,0].astype(int)
X = data[:,1:]

crime_lists = create_crime_lists(data, id_to_crime_dict)

# crime_lists = create_crime_lists(im.preprocess(im.vectorize(im.read('../data/train.csv'), ['time', 'latitude', 'longitude'])))
print("Wololo!")
for cl in crime_lists:
	print("foo")
	arr = np.asarray(crime_lists[cl])
	plot_crimes(arr, cl)