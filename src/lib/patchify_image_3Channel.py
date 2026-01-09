import numpy as np
from patchify import patchify

def patchify_image_3Channel(x, patch_dims):
    patches_img = patchify(x, (patch_dims, patch_dims, 3), step=patch_dims)  
    patches_flattened_img = np.squeeze(patches_img, axis = 2)
    patches_flattened_img = patches_img.reshape(-1, patch_dims, patch_dims,3)
    return(patches_flattened_img)