# -*- coding: utf-8 -*-
import numpy as np
def normalize_CNN_input_chN(z_stack_in, channel_names):

    idx_cBF = [i for i, s in enumerate(channel_names) if s.startswith('BF')]
    idx_cUNET = [i for i, s in enumerate(channel_names) if s.startswith('UNET')]
    if idx_cBF != [] and idx_cUNET != []:
        idx_cBF= idx_cBF[0]
        idx_cUNET = idx_cUNET[0]     
        z_stack_c = np.delete(z_stack_in,[idx_cBF, idx_cUNET], axis = 3)
    else:
       z_stack_c = z_stack_in.copy()        

    z_stack_c = z_stack_c - z_stack_c.min(axis=(1,2,3), keepdims=True) # min-max scale across channels to maintain relative intensity
    z_stack_c = z_stack_c / z_stack_c.max(axis=(1,2,3), keepdims=True)

    if idx_cBF != [] and idx_cUNET != []:
        z_stack_BF = z_stack_in.copy() # handle BF appart from fluorescence data as it's not fluorescence
        z_stack_BF = z_stack_BF[:,:,:,idx_cBF]
        z_stack_BF = np.expand_dims(z_stack_BF, axis=-1)
        z_stack_out = np.concatenate((z_stack_c,z_stack_BF), axis = -1)

        z_stack_UNET = z_stack_in[:,:,:,idx_cUNET] # append binary just in case
        z_stack_UNET = np.expand_dims(z_stack_UNET, axis=-1)
        z_stack_out = np.concatenate((z_stack_out,z_stack_UNET), axis = -1)
    else:
        z_stack_out = z_stack_c
    
    return(z_stack_out)
