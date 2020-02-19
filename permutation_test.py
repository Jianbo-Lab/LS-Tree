import numpy as np  
from scipy.stats import percentileofscore

def permutation_test(X, Y, n_iters = 1000):
	# whether mean of X is larger than mean of Y.
	observed_diff = np.mean(X) - np.mean(Y)
	Z = np.concatenate([X,Y])
	l = len(Z)
	l_X = len(X)
	null_diffs = []
	for i in range(n_iters):
		idx = np.random.permutation(l)
		new_Z = Z[idx]
		null_diff = np.mean(new_Z[:l_X]) - np.mean(new_Z[l_X:])
		null_diffs.append(null_diff)

	prop_below_observed = percentileofscore(null_diffs, observed_diff) / 100.0

	pval = 1 - prop_below_observed

	return pval