# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.models import load_model

# Function to run predictions on a specific GPU
def predict_on_gpu(gpu_id, model_path, data_chunk, batch_size = 15):
    with tf.device('CPU'):
        img_tensor = tf.convert_to_tensor(data_chunk)
    with tf.device(f'/gpu:{gpu_id}'):
        model = load_model(model_path)
    return model.predict(img_tensor, batch_size = batch_size)

