def eliminate_zeros_FF(array):
    from scipy.ndimage import distance_transform_edt

    # Find positions of zero and non-zero values
    zeros = (array == 0)
    non_zeros = (array != 0)

    # Get distances to the nearest non-zero and their indices
    distance, nearest_indices = distance_transform_edt(zeros, return_indices=True)

    # Replace zeros with the nearest non-zero value
    array[zeros] = array[tuple(nearest_indices[:, zeros])]
    return(array)