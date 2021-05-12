"""This file contains the different methods to preprocess the data for GANomaly
in Tensorflow.
Version: 1.0
Made by: Edgar Rangel
"""

import tensorflow as tf 

def normalize_accros_channels(data, label, mean, std):
    """Function that normalize the tensors in the last axis (channels) with mean and standard deviation 
    given.
    Args:
        data: A Tensor with to be normalized in the last axis (Tensor Instance).
        label: A Tensor that contains the true label for the dataset (Tensor Instance).
        mean: An integer specifying the mean to normalize the data (Integer).
        std: An integer specifying the standard deviation to normalize the data (Integer).
    """
    normalized = data - mean / std
    return normalized, label

def min_max_scaler(data, label, new_min, new_max):
    """Function that scale tensors using min max scaler formula letting the values between the new min and 
    max values given.
    Args:
        data: A Tensor with to be normalized in the last axis (Tensor Instance).
        label: A Tensor that contains the true label for the dataset (Tensor Instance).
        new_min: An integer specifying the new minimum value for the data (Integer).
        new_max: An integer specifying the new maximum value for the data (Integer).
    """
    data_std = (data - tf.reduce_min(data)) / (tf.reduce_max(data) - tf.reduce_min(data))
    data_scaled = data_std * (new_max - new_min) + new_min
    return data_scaled, label

def resize(data, label, size):
    """Function that resize a 3D tensor frame by frame or 2D tensor given a specified size.
    Args:
        data: A Tensor with 3 or 2 dimension to be resized (Tensor Instance).
        label: A Tensor that contains the true label for the dataset (Tensor Instance).
        size: A Tuple with 2 elements (height, width) respectively being the new size of the frames (Tuple).
    """
    return tf.image.resize(data, size), label

def get_center_of_volume(data, label, n_frames):
    """Function that get the center portion of 3D tensor (video) given the n frames.
    Args:
        data: A Tensor with 3 or 2 dimension to be resized (Tensor Instance).
        label: A Tensor that contains the true label for the dataset (Tensor Instance).
        n_frames: An integer specifying the quantity of frames to extract from data (Integer).
    """
    half = tf.math.floordiv(tf.shape(data)[0], 2)
    return data[half - n_frames//2 : half + n_frames//2], label

def undo_enumerate(i, xy):
    """Function that revert the enumerate method applied over a tf.data.Dataset.
    Args:
        i: An integer with the index of the data in the Dataset instance (Integer).
        xy: A Tuple of Tensor instances that contains the data and true label respectively (Tuple).
    """
    return xy[0], xy[1]