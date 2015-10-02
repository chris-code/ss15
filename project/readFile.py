#imports
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import sklearn.cross_validation as cv

"""
Method to read the single data file used for splitting the data

Throws: ValueError Exception
thrown if given data file could not be found.
Solved by reading a default document
"""
def readTrainingDocument(pathTraining="train.json"):
    try:
        trainRecipes = pd.read_json(pathTraining)
    except:
        print("Data could not be found. Reading train.json as default.")
        trainRecipes = pd.read_json('train.json')
    return trainRecipes

"""
Function to split the original training data into a training and a test sub set

Parameters: splitRatio
Indicates the ratio by which the training and test data should be split
returns the splitted train and test data
"""
def splitData(splitRatio):
    trainRecipes = readTrainingDocument('train.json')
    training, testing = cv.train_test_split(trainRecipes, test_size=splitRatio)
    training = pd.DataFrame(training)
    testing = pd.DataFrame(testing)
    return training, testing

"""
This procedure uses the original training and test data and converts them into Term frequency (tf) Inverse Document Frequency (idf) Tupels
to be able to fit Classifiers on it
"""
def processDocuments(split=False, splitRatio=0.25, extTrainingFile='', extTestFile=''):
    trainRecipes = readTrainingDocument("train_mod.json")
    if(split == False):
        if(extTrainingFile):
            ingredientsTraining = pd.Series.from_csv(extTrainingFile, encoding='utf-8')
        else:
            trainRecipes['ingredients_string'] = [' '.join(i).strip() for i in trainRecipes['ingredients']]
            ingredientsTraining = trainRecipes['ingredients_string']
        if(extTestFile):
            ingredientsTest = pd.Series.from_csv(extTestFile, encoding='utf-8')
        else:
            testRecipes = readTrainingDocument('test_mod.json')
            """extract ingredients for each recipe of the json file for both training and test data"""
            testRecipes ['ingredients_string'] = [' '.join(i).strip() for i in testRecipes ['ingredients']]
            """convert ingredients of recipes into a panda list containing the ingredients specific a recipe"""
            ingredientsTest = testRecipes ['ingredients_string']
        
        vectorTraining = TfidfVectorizer(stop_words='english')                      # initialize a Tfidf Vectorizer to determine the tf and idf
        vectorizedIngredients = vectorTraining.fit_transform(ingredientsTraining)   # train the vectorizer on the training ingredients to determine most common ingredients
        vectorizedIngredientsTest = vectorTraining.transform(ingredientsTest)       # evaluate important ingredients from the test data
        predictors_Training = vectorizedIngredients                                 # Make a copy of the different vectorized ingredients
        predictors_Test = vectorizedIngredientsTest
        # contains the cuisine for each recipe
        cuisine = trainRecipes['cuisine']
        return testRecipes, cuisine, predictors_Training, predictors_Test

    else:
        trainRecipes = readTrainingDocument("train_mod.json")
        if(extTrainingFile):
            ingredientsTraining = pd.Series.from_csv(extTrainingFile, encoding='utf-8')
        else:
            training, testing = splitData(splitRatio)
       
            training['ingredients_string'] = [' '.join(i).strip() for i in training['ingredients']]
            testing ['ingredients_string'] = [' '.join(i).strip() for i in testing ['ingredients']]
        
            ingredientsTraining = training['ingredients_string']
            ingredientsTesting  = testing ['ingredients_string']
        
            cuisinesTraining = training['cuisine']
            cuisinesTesting  = testing ['cuisine']
        
        vectorTraining = TfidfVectorizer(stop_words='english')
        vectorizedIngredients = vectorTraining.fit_transform(ingredientsTraining)
        vectorizedIngredientsTest = vectorTraining.transform(ingredientsTesting)
        
        predTraining = vectorizedIngredients
        predTesting = vectorizedIngredientsTest
       
        return predTraining, predTesting, cuisinesTraining, cuisinesTesting