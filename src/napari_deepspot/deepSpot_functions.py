"""
DeepSpot functions
====================
In this section you will find all information about functions relative to deepmeta's original code.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def enhance(obj, dataset):
    """
    Run inference.
    :param dataset: Imgs to segment
    :type dataset: np.array
    :param model_path: Path of model
    :type model_path: str
    :return: List of output binary masks
    :rtype: np.array
    """

    model = tf.keras.models.load_model(obj.model_path, custom_objects={'custom_loss': custom_loss})
    for img in dataset:
        img = np.expand_dims(img, axis=0)
        res = model.predict(img)
        res = res.reshape(256, 256)
        obj.viewer.add_image(res, name="enhanced")
    return res


def custom_loss(y_true, y_pred):
    loss_fn = keras.losses.BinaryCrossentropy()
    bce = loss_fn(y_true, y_pred)
    max_pred = tf.reduce_max(y_true)
    max_true = tf.reduce_max(y_pred)
    mse_max = (max_true - max_pred) ** 2
    loss_value = bce + mse_max
    return loss_value



def prepare_image(obj):
    img_list = []
    layers = obj.viewer.layers.copy()
    if len(layers)>0:
        for img in layers:
            print(img)
            image = img.data / 255
            image = (np.array(image) / np.amax(image)).reshape(256, 256, 1)
            img_list.append(image)
        return np.array(img_list)
    else:
        print("No image open")