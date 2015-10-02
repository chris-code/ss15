"""Read and normalize the supplied data, then save as csv.

This module reads json files and extracts the recipe ingredients. The ingredients are then normalized and cleaned before being saved in
csv files. This module exists to save conversion time for later runs.

Attributes
    normalizeToCSV:
    train = String to the path of the json file containing the training data
    test = String to the path of the json file containing the test data
    fileNameTraining = String of the desired file name for the normalized training data output
    fileNameTest = String of the desired file name for the normalized test data output
    type = string dictating the desired type. Available are CO (Concatenation only), Naive (naive normalization), NLTK (tagging using NLTK)"""

import IngredientManipulations
import pandas as pd

def normalizeToCsv(train="train.json", test="test.json", fileNameTraining='', fileNameTest='', type='CO'):
    
    """Create variables that include the training and the test data"""
    try:
        trainRecipes = pd.read_json(train)
    except ValueError:
        print 'File',train,'not found. Aborting.'
        return
    
    try:    
        testRecipes = pd.read_json(test)
    except ValueError:
        print 'File',test,'not found. Aborting.'
        return
    
    """extract ingredients for each recipe of the json file for both training and test data"""
    trainRecipes['ingredients_string'] = [i for i in trainRecipes['ingredients']]
    testRecipes['ingredients_string'] = [i for i in testRecipes['ingredients']]
    
    """convert the pandas series to lists"""
    ingredientsTrain = list(trainRecipes['ingredients_string'])
    ingredientsTest = list(testRecipes['ingredients_string'])
    
    """Normalize according to desired type"""
    """Don't normalize, just concatenate"""
    if type=='CO':
        
        print 'Concatenating without normalizing...',
        
        """Set file names for the output"""
        if not fileNameTraining: fileNameTraining = 'ingredientsTrainCO.csv'
        if not fileNameTest: fileNameTest = 'ingredientsTestCO.csv'
        
        """iterate over all ingredients and normalize them"""
        for i in range(len(ingredientsTrain)):
            ingredientsTrain[i] = IngredientManipulations.normalizeIngredientsCO(ingredientsTrain[i])
        print 'training set concatenation complete...',
        for i in range(len(ingredientsTest)):
            ingredientsTest[i] = IngredientManipulations.normalizeIngredientsCO(ingredientsTest[i])
        print 'test set concatenation complete...',
            
        print 'concatenation complete.'  
              
    elif type == 'Naive':
        
        print"Normalizing using Naive approach...",
        
        """Set file names for the output"""
        if not fileNameTraining: fileNameTraining = 'ingredientsTrainNaive.csv'
        if not fileNameTest: fileNameTest = 'ingredientsTestNaive.csv'
        
        """iterate over all ingredients and normalize them"""
        for i in range(len(ingredientsTrain)):
            ingredientsTrain[i] = IngredientManipulations.normalizeIngredientsNaive(ingredientsTrain[i])
        print "training set normalization complete...",
        for i in range(len(ingredientsTest)):
            ingredientsTest[i] = IngredientManipulations.normalizeIngredientsNaive(ingredientsTest[i])
        print "test set normalization complete...",
            
        print "naive Normalization complete."
        
    elif type == "NLTK":
        
        print "Normalizing using NLTK approach...",
        
        """Set file names for the output"""
        if not fileNameTraining: fileNameTraining = 'ingredientsTrainNLTK.csv'
        if not fileNameTest: fileNameTest = 'ingredientsTestNLTK.csv'
        
        """iterate over all ingredients and normalize them"""
        for i in range(len(ingredientsTrain)):
            ingredientsTrain[i] = IngredientManipulations.normalizeIngredientsNLTK(ingredientsTrain[i])
        print "training set normalization complete...",
        for i in range(len(ingredientsTest)):
            ingredientsTest[i] = IngredientManipulations.normalizeIngredientsNLTK(ingredientsTest[i])
        print "test set normalization complete...",
            
        print "NLTK Normalization complete."
        
    else:
        
        print "Normalization type not recognized. Defaulting to concatenation only...",
        
        """Set file names for the output"""
        if not fileNameTraining: fileNameTraining = 'ingredientsTrainCO.csv'
        if not fileNameTest: fileNameTest = 'ingredientsTestCO.csv'
        
        """iterate over all ingredients and normalize them"""
        for i in range(len(ingredientsTrain)):
            ingredientsTrain[i] = IngredientManipulations.normalizeIngredientsCO(ingredientsTrain[i])
        print "training set concatenation complete...",
        for i in range(len(ingredientsTest)):
            ingredientsTest[i] = IngredientManipulations.normalizeIngredientsCO(ingredientsTest[i])
        print "test set concatenation complete...",
            
        print "concatenation complete." 
            
    """join all the ingredients back into the pandas series"""
    testRecipes['ingredients_string'] = [' '.join(i).strip() for i in ingredientsTest]
    ingredientsTest = testRecipes['ingredients_string']
    trainRecipes['ingredients_string'] = [' '.join(i).strip() for i in ingredientsTrain]
    ingredientsTrain = trainRecipes['ingredients_string']

    """save all normalized ingredients to csv files"""
    ingredientsTrain.to_csv(fileNameTraining, encoding='utf-8')
    ingredientsTest.to_csv(fileNameTest, encoding='utf-8')