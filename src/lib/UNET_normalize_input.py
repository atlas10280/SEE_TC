# -*- coding: utf-8 -*-

import numpy as np

def UNET_normalize_input(img_x):
    img_x[np.where(img_x == 65535)] = 65534
    img_x = img_x+1
    img_x = np.log10(img_x)                   
    mean = np.mean(img_x)
    std_dev = np.std(np.float64(img_x), dtype=np.float64) # run as float64 to prevent overflow
    std_dev = np.float16(std_dev) # convert back to 16 for calculation
    img_x = (img_x-mean)/std_dev
    img_x = img_x + abs(np.min(img_x)) # set min to 0
    img_x = img_x / np.max(img_x) # re-scale for 0-1 range
    return(img_x)
