# -*- coding: utf-8 -*-

import sys
sys.path.append('/mnt/local/data2/Bootsma/2D_CTC/src/') 
import utils.CTC_2d_utils as ctc_utils
import preprocessing.CTC_2d_preprocessing as ctc_pp
import numpy as np
# FF is estimated using mini-batches, sampled from the full data, then averaged
def split_and_build_FF_FG(array, n_splits, ds_dim = 25, us_dim = 25):
    # Get the total number of samples (first axis)
    total_samples = array.shape[0]
    
    # Create an array of indices to split the data unevenly
    indices = np.array_split(np.arange(total_samples), n_splits)
    # Initialize a list to hold the average of each split
    poly_splits = []
    poly_splits_ds = []
    averaged_splits = []
    
    # Iterate over the split indices
    for split_indices in indices:
        # Select the subset of the array corresponding to the split indices
        subset = array[split_indices,...]
        
        # Compute the mean of the subset along axis 0 (which combines the first axis samples)
        FF_subset,avgDS_subset, poly_fit_i = ctc_utils.build_FF_kask(subset, ds_dim=ds_dim, us_dim=us_dim)
        
        # Append the averaged subset to the result list
        averaged_splits.append(avgDS_subset)
        poly_splits.append(FF_subset)
        poly_splits_ds.append(poly_fit_i)
    
    # Stack the averaged subsets to form the final output array
    result_avg = np.stack(averaged_splits, axis=0)
    result_poly = np.stack(poly_splits, axis=0)
    poly_splits_ds = np.stack(poly_splits_ds, axis = 0)
    return (result_poly, result_avg, poly_splits_ds)

