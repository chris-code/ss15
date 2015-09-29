import numpy as np
import importer as im
from sklearn.svm import SVC
import sklearn.cross_validation as cv

path = "../data/train.csv"
data = im.to_numpy_array(im.vectorize(im.read(path, 30000)))
X = data[:,2:4]
Y = data[:,1]

train_X, test_X, train_Y, test_Y = cv.train_test_split(X, Y, test_size=0.33)

for c in [10**p for p in range(-3,4)]:
	print c
	clf = SVC(C=c)
	clf.fit(train_X, train_Y)
	print (clf.score(test_X, test_Y))
