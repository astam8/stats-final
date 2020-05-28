import numpy as np
from scipy import stats

""" time to define a ton of fitness functions. Fitness = dissimilarity
Constraint: ALWAYS POSITIVE! don't forget this.

Ordered data differentials -- dissimilarity(X_1, X_2) = |X_1 - X_2| + |Y_1 - Y_2|

"""

# Creates two distinct clusters that are more in the shape of a cloud
def data_diff(x, ref):
    return np.sum(np.abs(x - ref))

# The following 2 methods move one point really far
def skewness_diff(x, ref):
    return np.abs(stats.skew(x[0]) - stats.skew(ref[0])) + np.abs(stats.skew(x[1]) - stats.skew(ref[1]))

# This one tends to cluster the remaining data points together
def kurt_diff(x, ref):
    return np.abs(stats.kurtosis(x[0]) - stats.kurtosis(ref[0])) + np.abs(stats.kurtosis(x[1]) - stats.kurtosis(ref[1]))

# Creates two very distinct clusters that are almost line
def power_diff(x, ref):
    return np.abs(stats.pearsonr(np.log(np.abs(x[0])), np.log(np.abs(x[1])))[0] - stats.pearsonr(np.log(np.abs(ref[0])), np.log(np.abs(ref[1]))))[0]