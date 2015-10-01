def logloss(predictions, truth):
	ll = 0.0
	for i in range(predictions.shape[0]):
		true_crime_id = truth[i]
		try:
			prob = predictions[i, true_crime_id]
		except IndexError:
			prob = 0
		prob = max( min(prob, 1 - 10**(-15)) , 10**(-15))
		ll += math.log(prob)
	return (-1.0) * ll / predictions.shape[0]