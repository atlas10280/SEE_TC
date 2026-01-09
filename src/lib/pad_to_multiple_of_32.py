import numpy as np
def pad_to_multiple_of_32(array):
    x_pad = (32 - array.shape[0] % 32) % 32
    y_pad = (32 - array.shape[1] % 32) % 32
    
    # Pad only on the right and bottom
    padded_array = np.pad(array, ((0, x_pad), (0, y_pad), (0, 0)), mode='constant', constant_values=0)
    
    return padded_array