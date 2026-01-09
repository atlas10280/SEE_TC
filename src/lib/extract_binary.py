import numpy as np
import nd2

# take an analyzed image and extract the binary mask based on assigned name
def extract_binary(file, binary_ID):
    print("Extracting binary layer: "+binary_ID)
    print(file)
    try:
        # with nd2.ND2File(file) as metadata:
        metadata = nd2.ND2File(file)
        bin_layers = metadata.binary_data
        metadata.close()
        bin_n_masks = len(bin_layers)
        for L in range(bin_n_masks): # identify the index of the 'cells' binary defined by Marina
            print("Looking for layer: "+binary_ID)
            bin_layer_i = bin_layers[L]# select a binary layer
            bin_layer_name_i = bin_layer_i.name
            if bin_layer_name_i == binary_ID:
                print("Layer found")    
                          
                #for the pre-defined binary layer in a file, extract a sub-region for training
                bin_layer_i = bin_layers[L]# select a binary layer
                #bin_layer_name_i = bin_layer_i[1]
                bin_layer_i = np.asarray(bin_layer_i)
                bin_layer_i[bin_layer_i>0] = 255 #re-scale binary to 0's and 255's
                bin_layer_i = np.expand_dims(bin_layer_i,2) # add a dimension for monochannel
                return(bin_layer_i)
                break
    except: 
        print("Binary layer not found")