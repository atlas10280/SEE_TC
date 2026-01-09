import numpy as np
from patchify import patchify

def patchify_image(x,patch_dims):
    patches_img = patchify(x, (patch_dims, patch_dims, 1), step=patch_dims)  
    patches_flattened_img = patches_img.reshape(-1, patch_dims, patch_dims)
    patches_flattened_img = np.expand_dims(patches_flattened_img,3)    
    return(patches_flattened_img)