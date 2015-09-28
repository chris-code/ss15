import numpy as np
import importer as im
from sklearn.svm import SVC
import sklearn.cross_validation as cv

# initialize counters, label vector Y and a dictionary to get the number representation of a class
counter = 0
Y = []
class_dict = {}
class_counter = 0

# count data points and create label vector Y
for c in im.read('../data/train.csv'):
	counter += 1
	if not c[1] in class_dict:
		class_dict[c[1]] = class_counter
		class_counter += 1
	Y.append(class_dict[c[1]])

# reverse class dict, to get string, which belongs to class number
reverse_class_dict = [(v,k) for (k,v) in class_dict.items()]

# initialize data points
X = np.empty((counter, 2))

# extract location data
for i,c in enumerate(im.read('../data/train.csv')):
	X[i][0] = c[7]
	X[i][1] = c[8]

train_X, test_X, train_Y, text_Y = cv.train_test_split(X, Y, test_size=0.33)
	
clf = SVC()
clf.fit(train_X, train_Y)
print (clf.score(test_X, test_Y))
