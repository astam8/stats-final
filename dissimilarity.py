import numpy as np
from scipy import stats

def data_diff(x, ref):
    return np.sum(np.abs(x - ref))

def skewness_diff(x, ref):
    return np.abs(stats.skew(x[0]) - stats.skew(ref[0])) + np.abs(stats.skew(x[1]) - stats.skew(ref[1]))

def kurt_diff(x, ref):
    return np.abs(stats.kurtosis(x[0]) - stats.kurtosis(ref[0])) + np.abs(stats.kurtosis(x[1]) - stats.kurtosis(ref[1]))

def power_diff(x, ref):
    return np.abs(stats.pearsonr(np.log(np.abs(x[0])), np.log(np.abs(x[1])))[0] - stats.pearsonr(np.log(np.abs(ref[0])), np.log(np.abs(ref[1]))))[0]