"""Normalize and clean Ingredients.

This module is used to normalize the Ingredients. It offers a naive way (normalizeIngredientsNaive) and
a way that uses nltk tagging (normalizeIngredientsNLTK), as well as a simple cleanup
in which the tokens of multi-token ingredients are concatenated with an underscore (normalizeIngredientsCO).
"""

from readFile import *
import nltk

def extractIngredientsFromRecipe(data, recipeNumber):
	"""Extract the ingredients from a recipe and return a list"""
	ingredients = (data[::][recipeNumber]['ingredients'][x] for x in range (0, len(data[::][recipeNumber]['ingredients'])))
	return list(ingredients)

def extractCuisineFromRecipe(data, recipeNumber):
	"""Extract and return the cuisine from a given recipe"""
	cuisine = (data[recipeNumber]['cuisine'])
	return (cuisine)

def normalizeIngredientsCO(ingredients):
	"""Concatenate the tokens of ingredients with multiple tokens"""
	
	"""data structures to hold ingredients with one or multiple tokens respectively"""
	oneToken = []
	moreTokens = []

	"""iterate over every ingredient in the recipe"""
	for ingredient in ingredients:
		"""check if the ingredient only has one token; if so, add to one token list; if not, add to multiple tokens list"""
		if ingredient == ingredient.split(" ")[0]:
			oneToken.append(ingredient)
		else:
			moreTokens.append(ingredient)
		
	"""all single token ingredients are added to the normalized Ingredients list"""
	normalizedIngredients = oneToken
			
	"""multiple token ingredients are added back to the normalized ingredient list as a string"""
	for ingredient in moreTokens:		
		normalizedIngredients.append('_'.join(ingredient.split(" ")))

	return normalizedIngredients

def normalizeIngredientsNaive(ingredients):
	"""Normalize the ingredients in a given list using a naive attempt"""
	
	"""data structures to hold ingredients with one or multiple tokens respectively"""
	oneToken = []
	moreTokens = []

	"""iterate over every ingredient in the recipe"""
	for ingredient in ingredients:
		"""check if the ingredient only has one token; if so, add to one token list; if not, add to multiple tokens list"""
		if ingredient == ingredient.split(" ")[0]:
			oneToken.append(ingredient)
		else:
			moreTokens.append(ingredient)
		
	"""Since they need no normalization, all single token ingredients are added to the normalized Ingredients list"""
	normalizedIngredients = oneToken
			
	"""all multiple token ingredients are split into their respective tokens"""
	for ingredient in moreTokens:
		normalizedIngredient = []
		ingredientTokenList = ingredient.split(" ")
	
		"""Those tokens are discarded if they look like an adjective or irrelevant addition; else they are kept""" 
		for token in ingredientTokenList:
			if token.endswith("ed"): pass
			elif token.endswith("ing"): pass
			elif token.endswith("boneless"): pass
			elif token.endswith("skinless"): pass
			else : normalizedIngredient.append(token)
			
		"""The normalized Ingredient is added back to the normalized Ingredient list as a string"""		
		normalizedIngredients.append('_'.join(normalizedIngredient))

	return normalizedIngredients

def normalizeIngredientsNLTK(ingredients):
	""""normalize the ingredients in a given list using nltk"""
	
	"""Data structure to store the normalized Ingredients"""
	normalizedIngredients=[]
	
	"""Clean and append every ingredient in the list"""
	for ingredient in ingredients:
		normalizedIngredients.append(cleanIngredient(ingredient))
		
	return normalizedIngredients
	
def cleanIngredient(ingredient):
	"""clean a given ingredient from unneeded parts using nltk"""
	
	"""data structure to hold the cleaned tokens"""
	cleaned_tokens = []
	
	"""tokenize the ingredient"""
	tokens = nltk.word_tokenize(ingredient)
	
	"""tag the ingredient"""
	tagged_tokens = nltk.pos_tag(tokens)
	
	"""iterate the tokens and remove unneeded words"""
	for i in range(0,len(tagged_tokens)):
		if ((tagged_tokens[i][1])!='JJ' and (tagged_tokens[i][1])!='VBD' and (tagged_tokens[i][1])!='VBG'): cleaned_tokens.append(tagged_tokens[i][0])
	
	"""check if any words made it through; if not, just use the original tokens"""
	if not cleaned_tokens:
		for i in range(0,len(tagged_tokens)): cleaned_tokens.append(tagged_tokens[i][0])
	
	""""create a new string to store the cleaned ingredient"""
	cleaned_ingredient = '_'.join(cleaned_tokens)
		
	return cleaned_ingredient

def normalizePandasNLTK(pandas_dataframe):
	"""normalize pandas ingredient series and returns a pandas ingredient series"""
	
	"""Convert Pandas dataframe to list"""
	df2list = list(pandas_dataframe['ingredients_string'])

	"""Iterate over all recipes and normalize the ingredients"""
	for i in range(len(df2list)):
		df2list[i] = normalizeIngredientsNLTK(df2list[i])

	
	result = [' '.join(i).strip() for i in ingredientsTest]
	
	return result