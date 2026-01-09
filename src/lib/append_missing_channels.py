# -*- coding: utf-8 -*-
import numpy as np
def append_missing_channels(img, channels_aquired, channels_expected = ['3', '4', '5', '6', '7', 'BF', 'UNET_open']):

    # channels_aquired = ['350x_455m', '560x_607m', '648x_684m', '740x_809m', 'BF', 'UNET_open']
    channels_aquired = {s[0] for s in channels_aquired} # Get the first character of each channel name

    insert_indices = [i for i, val in enumerate(channels_expected) if val[0] in channels_aquired]  # Find indices of channels not aquired

    z,x,y = img.shape[0:3] # Create new array filled with zeros
    new_arr = np.zeros((z, x, y, len(channels_expected)), dtype=img.dtype)

    # Step 3: Insert the existing channels into the correct positions
    for i, idx in enumerate(insert_indices):
        new_arr[..., idx] = img[..., i]
    return(new_arr)