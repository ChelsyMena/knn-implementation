[[00 Main - Introduction to Machine Learning]]
# KNN Project
#todo/Maestría/IML
## Code
- [x] function for the random selection, used for train test split and cross validation
- [x] function for the distance
- [x] function for the accuracy
- [x] function for the time spent on prediction
- [x] function for knn
	- [x] deal with cases where majority vote is tied: random, closest distance?
- [x] EDA:
	- [x] quick overviews of the dataset, mean and stdevs for the 21 features
	- [x] feature by feature distributions maybe, q-q plots to test normality
- [x] cross validation:
	- [x] how many folds? - search literature to justify
	- [x] tune the k in each fold?
	- [x] plots
		- [x] accuracy vs k
- [x] data reduction algorithms
	- [x] Bayesian region cleaning
		- [x] run the algorithm with the best k again and compare time and accuracy
	- [x] the one that removers the middle of the cluster 
		- [x] run the algorithm with the best k again and compare time and accuracy
- [ ] Methods for speeding up the calculation
	- [ ] precalculate the distance matrix
		- [ ] run the algorithm (with the original dataset) with the best k again and compare time and accuracy
		- [ ] run the algorithm (with the double cleaned dataset) with the best k again and compare time and accuracy
	- [x] KD Trees
- [ ] Imbalanced Dataset

## Report
- [ ] Abstract
- [ ] Introduction
- [ ] Methodology
- [ ] Results
- [ ] Conclusions
- [ ] References

## Notes 
Nice pairs of features for plotting
16 vs 6

Nice random seed 13, k = 47 and acc = 85%
random seed 33, k=40, acc 0.850250


random seed 77
   n_splits  Best k  Best Average Accuracy
0         3      61               0.856714
1         4      71               0.859250
2         5      86               0.859750
3        10     100               0.863250