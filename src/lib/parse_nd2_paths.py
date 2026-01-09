# -*- coding: utf-8 -*-

def list_files_recursive(path):
    import os
    file_list = []
    for root, _, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def parse_nd2_paths(path, model_ID, recursive = False):

    import os
    import fnmatch
    import pandas as pd

    if recursive == False:
        all_files = os.listdir(path) # List all files in the directory
    if recursive == True:
        all_files = list_files_recursive(path) # List all files in the directory
    # Use fnmatch to filter files based on the sub-pattern
    matching_files = [filename for filename in all_files if fnmatch.fnmatch(filename, f'*{model_ID}*')]
    return(matching_files)
    
