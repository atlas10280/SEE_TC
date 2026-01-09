# -*- coding: utf-8 -*-


import cv2
import numpy as np
# Used to downscale a 20x image to 10x, will work on other multiples as well
def downscale_by_factor_of(image, factor = 2):
    if image is None or not hasattr(image, 'shape'):
        raise ValueError("Invalid image input. Ensure the input is a valid NumPy array.")
    
    height, width = image.shape[:2]  # Extract height and width
    new_width = int(width // factor)  # Ensure dimensions are integers
    new_height = int(height // factor)
    
    if len(image.shape) == 2:  # Grayscale image
        # Resize directly for single-channel images
        downscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        # Handle multi-channel images
        channels = []
        for c in range(image.shape[2]):
            # Resize each channel independently
            resized_channel = cv2.resize(image[:, :, c], (new_width, new_height), interpolation=cv2.INTER_AREA)
            channels.append(resized_channel)
        
        # Stack the resized channels back together
        downscaled_image = np.stack(channels, axis=2)
    
    return downscaled_image