# -*- coding: utf-8 -*-


import numpy as np

def count_labels(y_pred):
    unique_vals, counts = np.unique(y_pred, return_counts=True)
    freq_dict = dict(zip(unique_vals, counts))
    print(freq_dict)