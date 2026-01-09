# # -*- coding: utf-8 -*-
# """
# Created on Thu Nov 16 13:07:21 2023

# @author: mbootsma
# """

# from skimage import measure
# import pandas as pd
import sys
import tifffile
import os
from scipy.ndimage import label
from skimage import measure
import nd2
import pandas as pd
# sys.path.append('/mnt/local/data2/Bootsma/2D_CTC/src/') 
# import utils.CTC_2d_utils as ctc_utils
sys.path.append('/mnt/local/data2/Bootsma/2D_CTC/src/analysis/publication_code/src/') 
import SEE_TC as ctc

def extract_region_props_LFI(img, binary, channel_IDs, verbose = True):
    """
    given a paired image and binary, extract the associated features

    The metadata must be produced by the user
    Channel_IDs should be provided and need to correspond to the channel order in which the tiff was written
    They will be used as column names for the intensity output values

    iteratively measure the pixel intensity of each channel for a given binary layer of a given image
    name the columns based on the channel ID
    merge channels
    measure the morphology and merge to pixel intensity data, then return
    """
    # ID number and name of lfi channels
    channel_count = len(channel_IDs)    # how many channels present?
    if verbose: 
        print("n channels: "+str(channel_count))
    flag = 0
    
    for i in range(0,channel_count):  
        channel_name_i = channel_IDs[i]  # exctract channel name associated with the current index
        # if channel_name_i == "BF":
        #     continue
        # print(channel_name_i)
        img_i = img[:,:,i].copy() #exctract channel image associated with the current index
        if verbose: 
            print(channel_name_i)
            print(img_i.shape, binary.shape)
        
        obj_props_fluor_i = measure.regionprops_table(binary,img_i, properties = [
            "label",    # sanity check, should match across rows
            "intensity_max",
            "intensity_mean",
            "intensity_min",
            "image_intensity"
        ])
        obj_props_fluor_i = pd.DataFrame(obj_props_fluor_i)
        
        obj_props_fluor_i = obj_props_fluor_i.rename(
            columns={
                "label": ("label" + "_" + channel_name_i),    
                "intensity_max": ("intensity_max" + "_" + channel_name_i),    
                "intensity_mean": ("intensity_mean" + "_" + channel_name_i),    
                "intensity_min": ("intensity_min" + "_" + channel_name_i),
                "image_intensity": ("image_intensity" + "_" + channel_name_i)
                                  }
            )
        if flag == 0:
            extracted_fluor = obj_props_fluor_i
            flag += 1
        else:
            extracted_fluor = extracted_fluor.join(obj_props_fluor_i)
    
    # extract the morphological measures and add these to the dataframe (only once, as not associated with fluorescence)
    obj_props_morph = measure.regionprops_table(binary, img_i, properties = [
        # additional region properties exist, this is where you would select or remove features
        "label",    # sanity check, should match across rows
        "coords",
        "bbox",
        "centroid",
        "area",
        "solidity",  
        "eccentricity",
        "extent",
    ])
    obj_props_morph = pd.DataFrame(obj_props_morph)
    
    extracted_region_props = extracted_fluor.join(obj_props_morph)
    
    return extracted_region_props 


# import xml.etree.ElementTree as ET

# def get_channel_names_tiff(tiff_path):
#     with tifffile.TiffFile(tiff_path) as tif:
#         # Parse the OME-XML metadata
#         ome_metadata = tif.ome_metadata
#         root = ET.fromstring(ome_metadata)

#         # Find all Channel names
#         namespaces = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}  # may need to adjust version
#         channels = root.findall('.//ome:Channel', namespaces)
#         channel_names = [ch.attrib.get('Name') for ch in channels]

#     return(channel_names)

# import os

# import nd2
# def extract_features_tiff(file_name_i):
#     # try:
        
#         tiff_name_i = file_name_i+".FFTTRB21.tiff"

#         print("======  ======  ======\n")
#         print("Processing "+file_name_i)

#         if os.path.isfile(tiff_name_i) == False:
#                 print("skipping sample ("+tiff_name_i+") no image")
#                 print("###############\n")     

#         img = tifffile.imread(tiff_name_i)
#         img = img.transpose(1,2,0)
#         print(img.shape)

#         c_names = get_channel_names_tiff(tiff_name_i)
#         print(c_names)

#         idx_bin = c_names.index("UNET_open")
#         binary = img[:,:,idx_bin]
#         print(binary.shape)
#         ######### Handle 20x
#         with nd2.ND2File(file_name_i+".nd2") as ndfile:
#                 text_unparsed = ndfile.text_info
#                 optics = text_unparsed['optics']

#         ########### check the magnification
#         if '10x' in optics:
#                 input_magnification = 10                        
#         elif '20x' in optics:
#                 input_magnification = 20
#                 scale_factor = input_magnification/10 # unet is trained at 10x so scale to fit that
#                 img = ctc.downscale_by_factor_of(img, scale_factor) # channel last
#                 binary = ctc.downscale_by_factor_of(binary, scale_factor) # channel last
#         else:
#                 print("skipping sample, unknown magnification")
#                 print("###############\n")            
#         print(img.shape, binary.shape)
#         #########

#         labeled_array, num_features = label(binary)
#         print("num_features="+str(num_features))
#         print(labeled_array.shape)
#         print("# EXTRACTING FEATURES #\n")        
#         features_i = extract_region_props_LFI(img, labeled_array, c_names) # extract all channels for a given binary
#         print("writing features\n")
#         features_i.to_csv(file_name_i+'_FFTTRBKask_region_features.tsv', sep = "\t", index = False)
#         return(features_i)
#     # except:
#     #     print("Unknown Error, skipping "+file_name_i)
#     #     print("======  ======  ======\n")

    
# def extract_features_dask_tiff(file_name_i):
# #     try:        
#         tiff_name_i = file_name_i+".FFTTRB21.dask.tiff"
#         print("======  ======  ======\n")
#         print("Processing "+file_name_i)

#         if os.path.isfile(tiff_name_i) == False:
#                 print("skipping sample ("+tiff_name_i+") no image")
#                 print("###############\n")     

#         img = tifffile.imread(tiff_name_i)
#         img = img.transpose(1,2,0)
#         print(img.shape)

#         c_names = get_channel_names_tiff(tiff_name_i)
#         print(c_names)

#         idx_bin = c_names.index("UNET_open")
#         binary = img[:,:,idx_bin]
#         print(binary.shape)
#         ######### Handle 20x
#         print(file_name_i+".nd2")
#         with nd2.ND2File(file_name_i+".nd2") as ndfile:
#                 text_unparsed = ndfile.text_info
#                 optics = text_unparsed['optics']

#         ########### check the magnification
#         if '10x' in optics:
#                 input_magnification = 10                        
#         elif '20x' in optics:
#                 input_magnification = 20
#         elif '40x' in optics:
#                 input_magnification = 40
#         else:
#                 input_magnification = -9
#                 print("skipping sample, unknown magnification")
#                 print("###############\n")   
#         print(input_magnification) 
#         if input_magnification > 0:
#             scale_factor = input_magnification/10 # unet is trained at 10x so scale to fit that
#             img = ctc.downscale_by_factor_of(img, scale_factor) # channel last
#             binary = ctc.downscale_by_factor_of(binary, scale_factor) # channel last        
#             print(img.shape, binary.shape)
#             #########
#             labeled_array, num_features = label(binary)
#             print(labeled_array.shape)
#             print("# EXTRACTING FEATURES #\n")        
#             features_i = extract_region_props_LFI(img, labeled_array, c_names) # extract all channels for a given binary
#             print("writing features\n")
#             features_i.to_csv(file_name_i+'_FFTTRBKask_region_features.tsv', sep = "\t", index = False)
#             return(features_i)
# #     except:
# #         print("Unknown Error, skipping "+file_name_i)
# #         print("======  ======  ======\n")

# def extract_features_dask_tiff_tile(file_name_i):
#     # try:
        
#         tiff_name_i = file_name_i+".FFTTRB21.dask.tiff"
#         img_path_prefix = file_name_i.split("_tile_")[0] if "_tile_" in file_name_i else file_name_i
#         nd2_path = img_path_prefix+".nd2"

#         print("======  ======  ======\n")
#         print("Processing "+file_name_i)

#         if os.path.isfile(tiff_name_i) == False:
#                 print("skipping sample ("+tiff_name_i+") no image")
#                 print("###############\n")     

#         img = tifffile.imread(tiff_name_i)
#         img = img.transpose(1,2,0)
#         print(img.shape)

#         c_names = get_channel_names_tiff(tiff_name_i)
#         print(c_names)

#         idx_bin = c_names.index("UNET_open")
#         binary = img[:,:,idx_bin]
#         print(binary.shape)
#         ######### Handle 20x
#         print(nd2_path)
#         with nd2.ND2File(nd2_path) as ndfile:
#                 text_unparsed = ndfile.text_info
#                 optics = text_unparsed['optics']

#         ########### check the magnification
#         if '10x' in optics:
#                 input_magnification = 10                        
#         elif '20x' in optics:
#                 input_magnification = 20
#         elif '40x' in optics:
#                 input_magnification = 40
#         else:
#                 input_magnification = -9
#                 print("skipping sample, unknown magnification")
#                 print("###############\n")   
#         print(input_magnification) 
#         if input_magnification > 0:
#             scale_factor = input_magnification/10 # unet is trained at 10x so scale to fit that
#             img = ctc.downscale_by_factor_of(img, scale_factor) # channel last
#             binary = ctc.downscale_by_factor_of(binary, scale_factor) # channel last        
#             print(img.shape, binary.shape)
#             #########
#             labeled_array, num_features = label(binary)
#             print(labeled_array.shape)
#             print("# EXTRACTING FEATURES #\n")        
#             features_i = extract_region_props_LFI(img, labeled_array, c_names) # extract all channels for a given binary
#             print("writing features\n")
#             features_i.to_csv(file_name_i+'_FFTTRBKask_region_features.tsv', sep = "\t", index = False)
#             return(features_i)
#     # except:
#     #     print("Unknown Error, skipping "+file_name_i)
#     #     print("======  ======  ======\n")

# # version for publish
def extract_features_biasCorrected_tiff(file_name_i):
#     try:        
        tiff_name_i = file_name_i+".bias_corrected.tiff"
        print("======  ======  ======\n")
        print("Processing "+file_name_i)

        if os.path.isfile(tiff_name_i) == False:
                print("skipping sample ("+tiff_name_i+") no image")
                print("###############\n")     

        img = tifffile.imread(tiff_name_i)
        img = img.transpose(1,2,0)
        print(img.shape)

        c_names = ctc.get_channel_names_tiff(tiff_name_i)
        print(c_names)

        idx_bin = c_names.index("UNET_open")
        binary = img[:,:,idx_bin]
        print(binary.shape)
        ######### Handle 20x
        print(file_name_i+".nd2")
        with nd2.ND2File(file_name_i+".nd2") as ndfile:
                text_unparsed = ndfile.text_info
                optics = text_unparsed['optics']

        ########### check the magnification
        if '10x' in optics:
                input_magnification = 10                        
        elif '20x' in optics:
                input_magnification = 20
        elif '40x' in optics:
                input_magnification = 40
        else:
                input_magnification = -9
                print("skipping sample, unknown magnification")
                print("###############\n")   
        print(input_magnification) 
        if input_magnification > 0:
            scale_factor = input_magnification/10 # unet is trained at 10x so scale to fit that
            img = ctc.downscale_by_factor_of(img, scale_factor) # channel last
            binary = ctc.downscale_by_factor_of(binary, scale_factor) # channel last        
            print(img.shape, binary.shape)
            #########
            labeled_array, num_features = label(binary)
            print(labeled_array.shape)
            print("# EXTRACTING FEATURES #\n")        
            features_i = ctc.extract_region_props_LFI(img, labeled_array, c_names) # extract all channels for a given binary
            print("writing features\n")
            features_i.to_csv(file_name_i+'.region_features.tsv', sep = "\t", index = False)
            return(features_i)