
def split_array_trainTest(img, binary, prop_cut):
    n_tiles = img.shape[0]
    split_idx = int(n_tiles*prop_cut)

    test_img = img[split_idx+1:n_tiles,:,:,:]
    test_binary = binary[split_idx+1:n_tiles,:,:,:]
    
    training_img = img[0:split_idx,:,:,:]
    training_binary = binary[0:split_idx,:,:,:]
    return(test_img, test_binary, training_img, training_binary)