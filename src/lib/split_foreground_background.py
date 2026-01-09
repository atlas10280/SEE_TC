# -*- coding: utf-8 -*-

import numpy as np

def split_foreground_background(image_tile, mask_tile):
    # Expand mask dimensions to match image dimensions
    mask_tile_expanded = np.expand_dims(mask_tile, axis=-1)

    # Create foreground and background arrays
    foreground = np.where(mask_tile_expanded == 1, image_tile, np.nan)
    background = np.where(mask_tile_expanded == 0, image_tile, np.nan)
    
    return foreground, background

