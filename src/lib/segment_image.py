# -*- coding: utf-8 -*-



import cv2
import numpy as np
from skimage import restoration
import sys
# sys.path.append('/mnt/local/data2/Bootsma/2D_CTC/src/') 
# import utils.CTC_2d_utils as ctc_utils
sys.path.append('/mnt/local/data2/Bootsma/2D_CTC/src/analysis/publication_code/src/') 
import SEE_TC as ctc


def segment_image(img_i, c, model_path, window_size=(32, 32), step_size=16, mode = "GPU", GPU_ids = ['3']):       
        img_i = img_i.astype(np.float32) # convert for processing
        # Clip each channel prior to FF estimation
        print("Clipping...") # clip outliers
        image_clipped = ctc.adaptive_clipping_with_iqr(img_i[..., c]) 
        
        print("Flattening...") # Estimate flat field for each clipped channel
        flat_field_c = ctc.estimate_flat_field(image_clipped)
        flat_field_c = flat_field_c/np.max(flat_field_c) # normalize the correction mask

        image_flat = img_i[..., c] / flat_field_c # APPLY FLAT FIELD CORRECTION HERE

        # Normalize corrected image to the original dtype range (e.g., 0-65535 for uint16)
        image_flat = (image_flat * 65535 / np.max(image_flat)).astype(np.uint16)

        print("Subtracting background...") # Subtract background
        x = image_flat
        background_x = restoration.rolling_ball(x, radius = 21) # radius should be ~2x object size...
        x_rb = x-background_x

        img_2_seg = np.stack((x_rb,x_rb,x_rb), axis = -1) # extract single sample
        print(img_2_seg.shape)    
        print(img_2_seg.dtype)

        ##### NORMALIZE #####   
        img_2_seg = ctc.UNET_normalize_input(img_2_seg) 
        ##### predict and threshold (only using Hoechst as this appears to work best)
        img_2_seg = ctc.pad_to_multiple_of_32(img_2_seg)
        # seg_stack[file_names[i]] = img_2_seg
        y_pred = ctc.UNET_slideAvg(img_2_seg, model_path, window_size, step_size, mode, GPU_ids)


        # threshold pixel probabilities    
        th,bin_i = cv2.threshold((y_pred*255).astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bin_i[bin_i>0] = 1
        return(bin_i)