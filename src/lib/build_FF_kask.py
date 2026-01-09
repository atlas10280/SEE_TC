# -*- coding: utf-8 -*-

import sys
sys.path.append('/mnt/local/data2/Bootsma/2D_CTC/src/') 
import utils.CTC_2d_utils as ctc_utils
import preprocessing.CTC_2d_preprocessing as ctc_pp
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import convolve
def build_FF_kask(input, ds_dim = 25, us_dim = 25, blur_iter_max = 20):
        
    input_z = input

    z_count = input_z.shape[0]
    input_ds = np.zeros((z_count,ds_dim,ds_dim))
    for z in range(z_count):
        input_ds_z = ctc_utils.downsample_with_nan_handling(input_z[z,:,:], (ds_dim, ds_dim), agg_func=np.nanmedian)
        input_ds[z,:,:] = input_ds_z

    p_fit_in = np.nanmedian(input_ds, axis=0)
    p_fit_in = ctc_utils.fill_nan_ds(p_fit_in)
    
    print("fitting polynomial...")
    poly_fit = ctc_pp.build_correction_mask(p_fit_in, 2)

    # Detect and remove outliers
    reference = poly_fit
    reference_broadcasted = np.broadcast_to(reference, input_ds.shape)
    difference = input_ds - reference_broadcasted
    squared_difference = difference ** 2
    mean_square_deviation = np.nanmean(squared_difference, axis=(1, 2))

    outliers_idx, outlier_rms = ctc_utils.find_outlier_indices(mean_square_deviation)

    # if outliers are present, remove them, re-fit, and check for any additional outliers to new fit
    input_ds_O = input_ds.copy()
    input_rmOutlier = input.copy() # used for qaqc to review the input tile associated with a downsample tile
    i = 0
    input_ds_avg = np.nanmean(input_ds_O, axis=0) # if no outliers, then this is our avg input
    while len(outliers_idx) > 0:
        print(input_ds_O.shape)
        print(str(len(outliers_idx))+" outliers detected, removing & re-fitting...")
        input_ds_O = np.delete(input_ds_O, outliers_idx, axis=0)
        input_rmOutlier = np.delete(input_rmOutlier, outliers_idx, axis=0)
        input_ds_avg = np.nanmean(input_ds_O, axis=0)
        input_ds_avg = ctc_utils.fill_nan_ds(input_ds_avg)
        poly_fit = ctc_pp.build_correction_mask(input_ds_avg, 3)

        # Detect and remove outliers
        reference = poly_fit
        reference_broadcasted = np.broadcast_to(reference, input_ds_O.shape)
        difference = input_ds_O - reference_broadcasted
        squared_difference = difference ** 2
        mean_square_deviation = np.nanmean(squared_difference, axis=(1, 2))

        outliers_idx, outlier_rms = ctc_utils.find_outlier_indices(mean_square_deviation)
        i+=1
        if(len(outliers_idx) == 0):
            print("No more outliers!")
            print(str(input_ds_O.shape[0])+" samples were used in fitting profile...")

    poly_fit_01 = poly_fit


    target_shape = (us_dim, us_dim)
    zoom_factors = (target_shape[0] / poly_fit_01.shape[0], target_shape[1] / poly_fit_01.shape[1])
    poly_fit_01 = zoom(poly_fit_01, zoom_factors, order=3)  # order=3 applies cubic interpolation
    input_ds_avg_zoom = zoom(input_ds_avg, zoom_factors, order=3)  # order=3 applies cubic interpolation
    return(poly_fit_01,input_ds_avg_zoom, poly_fit)
    
