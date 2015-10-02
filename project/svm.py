import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

import numpy    as np
import pandas   as pd
import readFile as rf
import sklearn.cross_validation as cv

def makeSubmission():
    """Call method to create tf idf data from the json files"""
    testRecipes, cuisine, predictors_Training, predictors_Test = rf.processDocuments()
    
    """initialize a SVM""" 
    linSVM = SVC(kernel = 'linear', C=1)
    
    """fit the svm to the data"""
    linSVM.fit(predictors_Training, cuisine)
    
    """predict the data"""
    predictions = linSVM.predict(predictors_Test)
    
    """store the predictions into a DataFrame"""
    testRecipes['cuisine'] = predictions
    
    """write results to csv file to be able to submit it to kaggle"""
    testRecipes[['id','cuisine']].to_csv('results.csv', index = False)
    
if __name__ == '__main__':
    makeSubmission()