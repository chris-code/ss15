import numpy as np
import importer as im
from sklearn.svm import SVC
import sklearn.cross_validation as cv

# load data
path = "../data/train.csv"
data = im.vectorize(im.read(path, 30000), ['latitude', 'longitude', 'time', 'day', 'month', 'year', 'day_of_week'])
crime_to_id_dict = data.next()
data = im.to_numpy_array(im.preprocess(data, 1, 2))
data = im.ensure_unit_variance(data, range(1,8))

# separate data in features and labels
Y = data[:,0].astype(int)
X = data[:,1:]

# split data in training data and test data
train_X, test_X, train_Y, test_Y = cv.train_test_split(X, Y, test_size=0.33)

# run svm for several values for C
for c in [0.001,100000]:
	print c
	
	# create SVM
	clf = SVC(C=c,kernel='linear')
	
	# fit SVM
	clf.fit(train_X, train_Y)
	
	# calculate predictions
	y_pred = clf.predict(test_X)
	
	# print first 30 predictions and the score
	print y_pred[:30]
	print (clf.score(test_X, test_Y))
