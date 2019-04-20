"""
utility.py
Zhihan Yang, April 20 2019, Carleton College
"""

import numpy as np
import keras


def from_img_paths(img_paths, target_size, preprocess='normalize'):
    """
    Load image arrays from a list of paths. Reshape and preprocess image arrays.

    :param img_paths: a list of paths of images
    :param target_size: a tuple of form (image_height, image_width)
    :param preprocess: the way in which image arrays are preprocessed
        if preprocess == 'normalize', then the image arrays are divided by 255.
        if preprocess == 'vgg16', then the image arrays are normalized channel-wise
    """
    if not isinstance(img_paths, list):
        img_paths = [img_paths]

    img_arrays = []
    for img_path in img_paths:
        img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        if preprocess == 'normalize':
            img_arrays.append(img_array / 255.)
        elif preprocess == 'vgg16':
            img_arrays.append(keras.applications.vgg16.preprocess_input(img_array))

    return np.vstack([np.expand_dims(img_array, axis=0) for img_array in img_arrays])


def enforce_list(*args):
    """Make sure that inputted objects are / become lists."""
    args_list = list(args)
    for i, arg in enumerate(args_list):
        if not isinstance(arg, list):
            args_list[i] = [arg]
    return args_list[0] if (len(args_list) == 1) else args_list


def to_indexable(feature_maps):
    """
    Make feature_maps indexable using layer index and image index.

    :param feature_maps: a list of feature maps
    """
    feature_maps_formatted = np.zeros((len(feature_maps), feature_maps[0].shape[0]), dtype=np.ndarray)
    for layer in range(len(feature_maps)):
        for image in range(feature_maps[layer].shape[0]):
            feature_maps_formatted[layer, image] = np.array(feature_maps[layer][image, :, :, :])

    return feature_maps_formatted
