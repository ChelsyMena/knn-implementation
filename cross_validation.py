#%%
# IML Project - MLDM
# Chelsy Mena & Serhii V. 

#%% LIBRARIES USED

import numpy as np

#%% LOAD DATA

data = np.genfromtxt(
	'Files\waveform.data',
	delimiter=','
	)

# %% TRAIN TEST SPLIT

test_index = list(np.random.randint(0, data.shape[0], 1000))
train_index = [x for x in list(range(data.shape[0])) if x not in test_index]

test_data = data[test_index]
train_data = data[train_index]

print(f"Data Size: {data.shape[0]} x {data.shape[1]}")
print(f"Test Size: {test_data.shape[0]} x {test_data.shape[1]}")
print(f"Train Size: {train_data.shape[0]} x {train_data.shape[1]}")

# %% K FOLD CROSS VALIDATION


# %% PREDICTION

def euclidian_distance(point_a, point_b):

	"""
	Calculate the Euclidian Distance for two points
	"""

	diffs = 0

	for i in range(point_a.shape[0]):
		diffs += (point_a[i] - point_b[i])**2
	
	distance = diffs**0.5

	return distance


predictions = []

# Pick a point from the test set
for point_predict in test_data:

	# Calculate all distances and store them 
	distances = []

	for point in train_data: #enumerate(train_data):

		distance = euclidian_distance(point_predict[:21], point[:21]) #point[1][:21])
		distances.append(distance)

	# Pick the sqrt(n) neighbors and cast the majority vote (k=64)
	k_nearest = np.argpartition(distances, 64)[:64]

	# Store the predictions vs the real values 
	k_nearest_labels = train_data[k_nearest][:, 21]
	values, counts = np.unique(k_nearest_labels, return_counts=True)
	label_index = np.where(counts==max(counts))
	majority_label = values[label_index]

	#print(f"The prediction was {majority_label[0]}, and the true value was {point_predict[21]}")

	# Store the predictions vs the true stuff
	predictions.append([majority_label[0], point_predict[21]])

# %% Accuracy
comparision = list(map(lambda x: 1 if x[0]==x[1] else 0, predictions))
acc = sum(comparision)/len(predictions)
print(f"Accuracy: {acc}")

