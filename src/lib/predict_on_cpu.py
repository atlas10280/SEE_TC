# -*- coding: utf-8 -*-


import tensorflow as tf
# Function to run predictions on a specified single GPU
def predict_on_cpu(model_path, data_chunk):
    with tf.device('CPU'):
        img_tensor = tf.convert_to_tensor(data_chunk)    
        model = tf.keras.models.load_model(model_path)
    return model.predict(img_tensor, batch_size = 16)

