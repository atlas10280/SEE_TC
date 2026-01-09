# -*- coding: utf-8 -*-

import numpy as np
def adaptive_clipping_with_iqr(arr):
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)
    IQR = Q3 - Q1
    low_threshold = Q1 - 1.5 * IQR
    high_threshold = Q3 + 1.5 * IQR
    clipped_arr = np.clip(arr, low_threshold, high_threshold)
    return clipped_arr

