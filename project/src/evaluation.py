import math

def logloss(predictions, truth):
	'''Calculates and returns the log loss of the predictions, given that truth contains the true labels.
	
	Predictions is a numpy array of shape (#samples, #crime_types), providing the probabilites for each type
	of crime as predicted for each sample. truth is a subscriptable, where the i-th component provides the
	column index for the column corresponding to the correct crime. If it indicates a non-existing column,
	all predictions for that sample are assumed to be false.
	
	The value is computed in accordance with
	https://www.kaggle.com/c/sf-crime/details/evaluation
	'''
	ll = 0.0 # log loss
	for i in range(predictions.shape[0]):
		true_crime_id = truth[i]
		try:
			prob = predictions[i, true_crime_id]
		except IndexError: # This can be neccessary if the test set contains crime types that were not in the training set
			prob = 0
		prob = max( min(prob, 1 - 10**(-15)) , 10**(-15)) # Avoid extreme values
		ll += math.log(prob)
	return (-1.0) * ll / predictions.shape[0]