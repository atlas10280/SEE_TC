# -*- coding: utf-8 -*-

import sys
# sys.path.append('/mnt/local/data2/Bootsma/2D_CTC/src/') 
# import utils.CTC_2d_utils as ctc_utils
sys.path.append('/mnt/local/data2/Bootsma/2D_CTC/src/analysis/publication_code/src/') 
import SEE_TC as ctc

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np



def UNET_slideAvg(image, model_path = None, window_size=(32, 32), step_size=16, mode = "CPU", GPU_ids=["3"], batch_size = 16):
    print("testing import")
    if model_path is None:
        raise ValueError("Please provide model path, (including filename of model)")
    # Initialize the output probability array and count array
    print("input shape:")
    print(image.shape)
    height, width, channels = image.shape
    prob_aggregate = np.zeros((height, width))
    count_aggregate = np.zeros((height, width))

    patches = []
    coordinates = []

    # Slide the window over the image to collect patches
    for y in range(0, height - window_size[1] + 1, step_size):
        for x in range(0, width - window_size[0] + 1, step_size):
            patch = image[y:y + window_size[1], x:x + window_size[0], :]
            patches.append(patch)
            coordinates.append((x, y))
    patches = np.array(patches)
    print("input patch shape (array): ")
    print(patches.shape)
    print(len(coordinates))
    # Check the number of GPUs defined
    num_gpus = len(GPU_ids)
    if mode == "GPU":
        if num_gpus == 1:
            # If only one GPU is defined, run the single GPU prediction using the specified GPU
            gpu_id = GPU_ids[0]  # Use the first (and only) GPU ID
            print(f"Running on a single GPU: {gpu_id}")
            final_predictions = ctc.predict_on_single_gpu(model_path, patches)
        else:
            # If more than one GPU is defined, run multi-GPU predictions
            print(f"Running on {num_gpus} GPUs: {GPU_ids}")
            data_chunks, chunk_indices = ctc.split_data(patches, num_gpus)
            # Predict using multiple GPUs
            predictions = [None] * num_gpus  # Initialize a list to hold predictions in the correct order
            with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                future_to_index = {
                    executor.submit(ctc.predict_on_gpu, gpu_id, model_path, data_chunk, batch_size = batch_size): i
                    for i, (gpu_id, data_chunk) in enumerate(zip(GPU_ids, data_chunks))
                }
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        pred = future.result()
                        print(f"Predictions from GPU {GPU_ids[index]} have shape: {pred.shape}")
                        predictions[index] = pred
                    except Exception as exc:
                        print(f"GPU {GPU_ids[index]} generated an exception: {exc}")

            # Reorder the predictions to match the original order of patches
            final_predictions = np.empty_like(patches)
            for chunk_pred, indices in zip(predictions, chunk_indices):
                final_predictions[indices] = chunk_pred
    else:
        # If only one GPU is defined, run the single GPU prediction using the specified GPU
        print(f"Running on a single CPU")
        final_predictions = ctc.predict_on_cpu(model_path, patches)

    print(final_predictions.shape)

    # Aggregate the probabilities
    for (x, y), patch_prob in zip(coordinates, final_predictions):
        patch_prob_avg = np.mean(patch_prob, axis=-1)  # Average across the last dimension (channels)
        prob_aggregate[y:y + window_size[1], x:x + window_size[0]] += patch_prob_avg
        count_aggregate[y:y + window_size[1], x:x + window_size[0]] += 1
        

    # Average the probabilities
    count_aggregate[count_aggregate == 0] = 1  # if input isn't // by tile size then just set as 1 to avoid issues
    averaged_probabilities = prob_aggregate / count_aggregate

    return averaged_probabilities


