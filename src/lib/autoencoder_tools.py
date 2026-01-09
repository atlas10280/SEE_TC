# -*- coding: utf-8 -*-
"""
Created on Thu May 20 2024

@author: mbootsma
"""

import numpy as np
def preprocess_autoEncoder_01(ndarray):
    ##### Handle outliers
    # Compute the median and interquartile range (IQR)
	median = np.median(ndarray, axis=(0, 1, 2), keepdims=True)
	q1 = np.percentile(ndarray, 25, axis=(0, 1, 2), keepdims=True)
	q3 = np.percentile(ndarray, 75, axis=(0, 1, 2), keepdims=True)
	iqr = q3 - q1
	# Define threshold for outliers
	lower_bound = median - 1.5 * iqr
	upper_bound = median + 1.5 * iqr
	# Clip the outliers
	ndarray_clipped = np.clip(ndarray, lower_bound, upper_bound)
	# Normalize the data using the mean and standard deviation of the clipped data
	mean = np.mean(ndarray_clipped, axis=(0, 1, 2), keepdims=True)
	std = np.std(ndarray_clipped, axis=(0, 1, 2), keepdims=True)
	std[std==0]+=1
	# Normalize the data
	normalized_ndarray = (ndarray_clipped - mean) / std
	min_val = np.min(normalized_ndarray)
	max_val = np.max(normalized_ndarray)
	scaled_ndarray = (normalized_ndarray - min_val) / (max_val - min_val)
	return(scaled_ndarray)
