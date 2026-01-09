# -*- coding: utf-8 -*-


import cv2
def estimate_flat_field(image_channel, blur_radius=125):
    # Ensure kernel size is a positive odd integer
    kernel_size = int(blur_radius) * 2 + 1
    blurred = cv2.GaussianBlur(image_channel, (kernel_size, kernel_size), 0)
    return blurred

