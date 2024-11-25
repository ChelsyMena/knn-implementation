#%%
# IML Project - MLDM
# Chelsy Mena & Serhii V. 
# Data Reduction Functions

#%% LIBRARIES USED
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from cross_validation import knn_predictions

#%% LOAD DATA
data = np.genfromtxt('Files\waveform.data',	delimiter=',')

def bayesian_region(data):

	index_split = np.random.randint(0, data.shape[0])
	S1 = data[: index_split]
	S2 = data[index_split+1:]

	preds, _ = knn_predictions(S2, S1, 1)

	missclassified = []
	for pred in range(len(preds)):
		if preds[pred][0] != preds[pred][1]:
			missclassified.append(pred)
	keep = [x for x in list(range(len(preds))) if x not in missclassified]
	S1 = S1[keep]

	# Now backwards
	preds, _ = knn_predictions(S1, S2, 1)

	missclassified = []
	for pred in range(len(preds)):
		if preds[pred][0] != preds[pred][1]:
			missclassified.append(pred)
	keep = [x for x in list(range(len(preds))) if x not in missclassified]
	S2 = S2[keep]

	kept_data = np.vstack((S1, S2))

	return kept_data

def condensed(data):

	storage = [np.random.randint(0, data.shape[0])]
	dustbin = []

	for point in range(data.shape[0]):

		# Predict the point with the data in storage
		_, acc = knn_predictions(data[storage], data[[point]], 1)

		if acc == 1:
			dustbin.append(point)
		else:
			storage.append(point)
	
	return storage
# %%
