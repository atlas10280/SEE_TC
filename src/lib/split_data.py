import numpy as np
# Split the input data into chunks for each GPU
def split_data(data, num_splits):
    splits = np.array_split(data, num_splits)
    indices = np.array_split(np.arange(len(data)), num_splits)
    return splits, indices