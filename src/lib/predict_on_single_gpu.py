# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.models import load_model

def predict_on_single_gpu(model, data_chunk, batch_size = 16):
    pred = model.predict(data_chunk, batch_size = batch_size)
    return pred
