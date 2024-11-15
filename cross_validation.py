#%%
# IML Project - MLDM
# Chelsy Mena & Serhii V. 

#%% LIBRARIES USED
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% LOAD DATA
data = np.genfromtxt('Files\waveform.data',	delimiter=',')

# %% TRAIN TEST SPLIT

def split_data(data, n):
	"""
	Split data in two groups, one with n elements and another with the rest
	
	Input: numpy array dataset, dims a x b
	Output: two numpy arrays, dims b x n and b x (b-n)
	"""

	if n >= data.shape[0]:
		raise IndexError("The dataset has less examples than you think man. Pick a smaller n.")

	np.random.seed(7)
	n_index = list(np.random.choice(data.shape[0], 1000, replace=False))
	remaining_index = [x for x in list(range(data.shape[0])) if x not in n_index]

	n_data = data[n_index]
	remaining_data = data[remaining_index]

	return n_data, remaining_data

test_data, train_data = split_data(data, n=1000)

print(f"Data Size: {data.shape[0]} x {data.shape[1]}")
print(f"Test Size: {test_data.shape[0]} x {test_data.shape[1]}")
print(f"Train Size: {train_data.shape[0]} x {train_data.shape[1]}")

# %% K FOLD CROSS VALIDATION

def euclidian_distance(point_a, point_b):

	"""
	Calculate the Euclidian Distance for two points
	
	Input: Two points, numpy arrays
	Output: The distance between the two points
	"""

	diffs = 0

	for i in range(point_a.shape[0] - 1):
		diffs += (point_a[i] - point_b[i])**2
	distance = diffs**0.5

	return distance


def distance_matrix(train_data, test_data):

	"""
	Precalculate the distance matrix between two tables
	
	Input: Two numpy arrays with points
	Output: Distance matrix between the two, numpy array
	"""
	
	distance_matrix = []

	for test_point in test_data:
		row = []
		for train_point in train_data:
			distance = euclidian_distance(test_point, train_point)
			row.append(distance)
		distance_matrix.append(row)

	return np.array(distance_matrix)


def knn_predictions(train_data, test_data, k, distance_matrix=None):

	"""Predict the labels for a dataset
	
	Input: 
		- Train dataset, numpy array
		- Test dataset, numpy arrya
		- Number of neighbors to consider
		- Distance matrix if it's precalculated outside
	
	Output: 
		- Array with the predictions
		- Accuracy across the test set
	"""

	predictions = []

	# Pick a point from the test set
	for i in range(test_data.shape[0]):

		#If I precalculated the distances, just have to get the row
		if distance_matrix is not None:
			distances = distance_matrix[i]

		# Else, calculate all distances and store them
		else:
			distances = []

			for train_point in train_data:

				distance = euclidian_distance(test_data[i], train_point)
				distances.append(distance)

		# Pick the indexes of the k nearest neighbors
		k_nearest = np.argpartition(distances, k)[:k]
		k_nearest_labels = train_data[k_nearest][:, 21]

		# Cast simple majority vote. DOESNT TAKE TIES INTO ACCOUNT
		values, counts = np.unique(k_nearest_labels, return_counts=True)
		label_index = np.where(counts==max(counts))
		majority_label = values[label_index]

		# Store the predictions vs the true stuff
		predictions.append([majority_label[0], test_data[i][21]])

	# Calculate the accuracy
	comparision = list(map(lambda x: 1 if x[0]==x[1] else 0, predictions))
	accuracy = sum(comparision)/len(comparision)

	return predictions, accuracy

#%%
n_folds = 5
n_items = int(train_data.shape[0]/n_folds)

accuracies = []

for i in range(1, n_folds+1):

	# Split the small validation data
	val_index = range((i-1)*n_items, i*n_items)
	train_index = [x for x in list(range(train_data.shape[0])) if x not in val_index]

	fold_train = train_data[train_index]
	fold_test = train_data[val_index]

	calculated_distance_matrix = distance_matrix(fold_train, fold_test)

	accs = []
	for k in range(1, 100):
		_, acc = knn_predictions(fold_train, fold_test, k, calculated_distance_matrix)
		accs.append(acc)
	print(f"Done {i}")

	accuracies.append(accs)

# %% PLOT THE ACCURACIES

plt.figure(figsize=(10, 6))

# Loop over each list of accuracies
for i, line in enumerate(accuracies):

    sns.lineplot(x=range(1, len(line)+1), y=line, label=f'Fold {i+1}')

    # Find the index and value of the maximum point in the line
    max_index = np.argmax(line)
    max_value = line[max_index]
    plt.plot(max_index + 1, max_value, 'ro')

    # Annotate the maximum value with the index
    plt.text(max_index + 1, max_value, f'{max_index+1} / acc {round(max_value,3)*100}%', color='red', ha='center', va='bottom')

plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Best Number of Neighbors in Cross Validation')
plt.legend()
plt.show()
# %%
