import numpy as np

def thin_background_rankedBackground(x,y, background_proportion):
    #calculate the max value of each binary mask
    y_max = []
    for tile_i in range(y.shape[0]):
        y_max.append(np.mean(y[tile_i,:,:,:]))
    indices_cells = np.nonzero(y_max)[0]
    indices_background = np.where(np.array(y_max) == 0)[0]
    
    
    #find the average intensity of the images with no cells
    background_avgs = []
    background_medians = []
    for bg_idx in indices_background:
        background_avgs.append(np.mean(x[bg_idx,:,:,:]))
        background_medians.append(np.median(x[bg_idx,:,:,:]))
    import pandas as pd
    background_df = pd.DataFrame({
        'background_indices':indices_background,
        'avg_values':background_avgs,
        'median_values':background_medians
        })
    background_df = background_df.sort_values(by = 'avg_values', ascending=False)    
    
    # use all positive cell images, and add back n% of the background images, randomly sampling the images
    n_indices = int(np.round(len(indices_cells)*background_proportion)-1) # define how many background slides to include
    if n_indices > len(indices_background):
       n_indices = len(indices_background) 
       print("Background images exhausted, all "+str(n_indices-1)+" background tiles will be used")
    n_indices = n_indices-1
    indices_background = background_df.loc[range(n_indices),['background_indices']]
    indices_background = indices_background.to_numpy()
    indices_background = indices_background[:,0]
    # select the image and masks with/without background for final fitting
    image_dataset = np.concatenate([x[indices_cells,:,:,:], x[indices_background,:,:,:]], axis=0)
    mask_dataset = np.concatenate([y[indices_cells,:,:], y[indices_background ,:,:]], axis=0)
    mask_dataset = mask_dataset /255. # re-scale binary data to 0's and 1's
    return image_dataset, mask_dataset

