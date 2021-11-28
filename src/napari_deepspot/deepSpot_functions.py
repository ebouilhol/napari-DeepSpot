"""
DeepSpot functions
====================
In this section you will find all information about functions relative to deepmeta's original code.
"""

import numpy as np
from pathlib import Path
from appdirs import user_config_dir
from configparser import ConfigParser
from scipy import ndimage
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import cv2
import os
import skimage.measure as measure
import skimage.exposure as exposure



def enhance(dataset, model_path):
    """
    Run inference.
    :param dataset: Imgs to segment
    :type dataset: np.array
    :param model_path: Path of model
    :type model_path: str
    :return: List of output binary masks
    :rtype: np.array
    """

    model = tf.keras.models.load_model(model_path, custom_objects={'custom_loss': custom_loss})
    print("-"*100)
    print(dataset.shape)
    dataset = np.expand_dims(dataset, axis=0)
    print(dataset.shape)
    res = model.predict(dataset)
    print(res.shape)

    return res.reshape(256,256,1)


def custom_loss(y_true, y_pred):
    loss_fn = keras.losses.BinaryCrossentropy()
    bce = loss_fn(y_true, y_pred)
    max_pred = tf.reduce_max(y_true)
    max_true = tf.reduce_max(y_pred)
    mse_max = (max_true - max_pred) ** 2
    loss_value = bce + mse_max
    return loss_value


def get_images(path_list):
    import skimage.io as io
    dataset = []
    for file in path_list:
        try:
            img = io.imread(file)
            img = np.array(img, dtype=np.float32)
            assert np.amax(img) > 0
            assert img.shape[0] == 256
            assert img.shape[1] == 256
            dataset.append((np.array(img) / np.amax(img)).reshape(256, 256, 1))

        except Exception as e:
            print("Image {} not found.\n{}".format(file, e))
    assert len(dataset) != 0
    assert np.amax(dataset[0]) > 0
    return np.array(dataset)


def clean_layers(obj, vol_id=5):
    if len(obj.viewer.layers) != 0:
        while obj.viewer.layers:
            obj.viewer.layers.pop()
        try:
            obj.layout().itemAt(vol_id).widget().setParent(None)
        except:
            print("no volume displayed")

def prepare_image(obj):
    image = None
    if len(obj.viewer.layers) == 1:
        image = obj.viewer.layers[0].data / 255
        image = (np.array(image) / np.amax(image)).reshape(256, 256, 1)
        clean_layers(obj)
        obj.viewer.add_image(image, name="original")
    else:
        print("You do not have only one image opened.")
    return image