import numpy as np
import importer as im
from sklearn.svm import SVC
import sklearn.cross_validation as cv

path = "../data/train.csv"
#data = im.to_numpy_array(im.preprocess(im.vectorize(im.read(path, 30000))))

data = im.vectorize(im.read(path, 30000), ['latitude', 'longitude', 'time', 'day', 'month', 'year', 'day_of_week'])
crime_to_id_dict = data.next()
data = im.to_numpy_array(im.preprocess(data, 1, 2))
data = im.ensure_unit_variance(data)

Y = data[:,0].astype(int)
X = data[:,1:]

train_X, test_X, train_Y, test_Y = cv.train_test_split(X, Y, test_size=0.33)

for c in [1.5]:
	print c
	clf = SVC(C=c,gamma=10)
	clf.fit(train_X, train_Y)
	y_pred = clf.predict(test_X)
	print y_pred[:30]
	
	print (clf.score(test_X, test_Y))
