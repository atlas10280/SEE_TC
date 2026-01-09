# # -*- coding: utf-8 -*-
# """
# Created on Thu Nov 16 13:07:21 2023

# @author: mbootsma
# """

import numpy as np
import nd2

def contains_substring(string_list, search_string):
    '''
    Search a list of strings for a complete or partial match of a given pattern
    
    Parameters
    ----------
    string_list : list
        list of strings to query
    search_string: str
        single string to check the presence of in the string_list

    Returns
    -------
    bool
        
    '''
    for s in string_list:
        if search_string in s:
            return True
    return False

# if image has repeated channel names, this function will define them based on their sequential position in the input
def update_duplicates_with_suffix(arr):
    from collections import defaultdict
    # Dictionary to track occurrences of each string
    counts = defaultdict(int)
    
    # Iterate over the array and update duplicates with a suffix
    for i, value in enumerate(arr):
        counts[value] += 1
        if counts[value] > 1:
            arr[i] = f"{value}_{counts[value] - 1}"
    return arr

def extract_all_array_nd2(file):
    '''
    Take a multi-channel .nd2 image and extract the array for every channel(s) within
    
    Parameters
    ----------
    file : str
        filename of the .nd2 image you wish to extract data from.
    
    Returns
    -------
    img : np.ndarray
        image data for the channels of interest.
    
    channels : 
    '''
    
    print("Extracting image array:")
    print(file)
    img = nd2.imread(file)              # read full ND2 file
    with nd2.ND2File(file) as m:        # read associated metadata
        c_names = m._channel_names      # extract channel names
    print("Detected channels include:")
    c_names = update_duplicates_with_suffix(c_names)
    print(c_names)

    idx_l = [] # initialize index vector
    for channel_i in c_names: # for each channel of interest, determine the index in the nd2 array
        if contains_substring(c_names, channel_i) == False: # stop if invalid channel query pattern provided
            print("Error: query channel pattern not found ("+channel_i+")")
            break
                
        idx_i = [i for i, s in enumerate(c_names) if channel_i in s]
        # print(idx_i)
        idx_l.append(idx_i[0])
    
    img = img[idx_l,:,:] # extract the channels of interest as an ndarray
    img = np.squeeze(img)
    return [img,c_names]

# def extract_array_nd2(file, channels):
#     '''
#     Take a multi-channel .nd2 image and extract the array for specific channel(s) based on the name(s)
    
#     Parameters
#     ----------
#     file : str
#         filename of the .nd2 image you wish to extract data from.
#     channels : list
#         list of the channel(s) you wish to extract, will return in order of provided list.
#         partial matching is supported, e.g., a channel called "350x_390m" can be indexed by providing "350"

#     Returns
#     -------
#     img : np.ndarray
#         image data for the channels of interest.
        
#     '''
    
#     print("Extracting image array:")
#     print(file)
#     img = nd2.imread(file)              # read full ND2 file
#     with nd2.ND2File(file) as m:        # read associated metadata
#         c_names = m._channel_names      # extract channel names
#     print("Detected channels include:")
#     print(c_names)
    
#     idx_l = [] # initialize index vector
#     for channel_i in channels: # for each channel of interest, determine the index in the nd2 array
    
#         if contains_substring(c_names, channel_i) == False: # stop if invalid channel query pattern provided
#             print("WARNING: query channel pattern not found ("+channel_i+")")
#             continue
                
#         idx_i = [i for i, s in enumerate(c_names) if channel_i in s]
#         idx_l.append(idx_i)
    
#     img = img[idx_l,:,:] # extract the channels of interest as an ndarray
#     img = np.squeeze(img)
#     return img

# def extract_3D_array_nd2(file, channels):
#     '''
#     Take a multi-channel .nd2 image with multiple Z-layers and extract the array for specific channel(s) based on the name(s)
    
#     Parameters
#     ----------
#     file : str
#         filename of the .nd2 image you wish to extract data from.
#     channels : list
#         list of the channel(s) you wish to extract, will return in order of provided list.
#         partial matching is supported, e.g., a channel called "350x_390m" can be indexed by providing "350"

#     Returns
#     -------
#     img : np.ndarray
#         image data for the channels of interest.
        
#     '''
    
#     print("Extracting image array:")
#     print(file)
#     img = nd2.imread(file)              # read full ND2 file
#     with nd2.ND2File(file) as m:        # read associated metadata
#         c_names = m._channel_names      # extract channel names
#     print("Detected channels include:")
#     print(c_names)
    
#     idx_l = [] # initialize index vector
#     for channel_i in channels: # for each channel of interest, determine the index in the nd2 array
    
#         if contains_substring(c_names, channel_i) == False: # stop if invalid channel query pattern provided
#             print("Error: query channel pattern not found ("+channel_i+")")
#             break
                
#         idx_i = [i for i, s in enumerate(c_names) if channel_i in s]
#         idx_l.append(idx_i)
    
#     img = img[:,idx_l,:,:] # extract the channels of interest as an ndarray
#     img = np.squeeze(img, axis = 2)
#     return img
