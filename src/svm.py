import numpy as np
import importer as im
import evaluation as ev
import data_processing as dp
from sklearn.svm import SVC
import sklearn.cross_validation as cv

# load data
path = "../data/train.csv"
data = im.read_labeled(path, 10000)
data = dp.vectorize(data, 1, features=[('latitude', 7), ('longitude', 8), ('day', 0), ('day_of_week', 0), ('time', 0)])
crime_to_id_dict = data.next()
data = im.to_numpy_array(data)
data = dp.ensure_unit_variance(data, columns_to_normalize=(0,1,2,3,4))

# separate data in features and labels
Y = data[:,5].astype(int)
X = data[:,:5]

# split data in training data and test data
train_X, test_X, train_Y, test_Y = cv.train_test_split(X, Y, test_size=0.33)

# run svm for several values for C
for c in [0.1, 1, 1.5, 10,20,50,100,200]:
	print "C = {0}".format(c)
	
	# create SVM
	clf = SVC(C=c,kernel='rbf', gamma=1000)
	
	# fit SVM
	clf.fit(train_X, train_Y)

	# calculate predictions
	y_pred = clf.predict(test_X)
	
	# print first 30 predictions and the score
	print y_pred[:30]
	print (clf.score(test_X, test_Y))
