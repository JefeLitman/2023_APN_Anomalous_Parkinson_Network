"""This file contains the different methods to preprocess the data for GANomaly
in Tensorflow.
Version: 1.2
Made by: Edgar Rangel
"""

import tensorflow as tf 

def normalize_accros_channels(data, label, mean, std):
    """Function that normalize the tensors in the last axis (channels) with mean and standard deviation 
    given.
    Args:
        data (Tensor): A Tensor with to be normalized in the last axis.
        label (Tensor): A Tensor that contains the true label for the dataset.
        mean (Decimal): An integer specifying the mean to normalize the data.
        std (Decimal): An integer specifying the standard deviation to normalize the data.
    """
    normalized = data - mean / std
    return normalized, label

def min_max_scaler(data, label, new_min, new_max):
    """Function that scale tensors using min max scaler formula letting the values between the new min and 
    max values given.
    Args:
        data (Tensor): A Tensor with to be scaled in the last axis.
        label (Tensor): A Tensor that contains the true label for the dataset.
        new_min (Decimal): An integer specifying the new minimum value for the data.
        new_max (Decimal): An integer specifying the new maximum value for the data.
    """
    data_std = (data - tf.reduce_min(data)) / (tf.reduce_max(data) - tf.reduce_min(data))
    data_scaled = data_std * (new_max - new_min) + new_min
    return data_scaled, label

def resize(data, label, size):
    """Function that resize a 3D tensor frame by frame or 2D tensor given a specified size.
    Args:
        data (Tensor): A Tensor with 4 or 3 dimension to be resized.
        label (Tensor): A Tensor that contains the true label for the dataset.
        size (Tuple): A Tuple with 2 elements (height, width) respectively being the new size of the frames.
    """
    return tf.image.resize(data, size), label

def get_center_of_volume(data, label, n_frames):
    """Function that get the center portion of 3D tensor (video) given the n frames.
    Args:
        data (Tensor): A Tensor with 4 dimension to be cropped with (frames, height, width, channels) shape.
        label (Tensor): A Tensor that contains the true label for the dataset.
        n_frames (Integer): An integer specifying the quantity of frames to extract from data.
    """
    half = tf.math.floordiv(tf.shape(data)[0], 2)
    return data[half - n_frames//2 : half + n_frames//2], label

def undo_enumerate(i, xy):
    """Function that revert the enumerate method applied over a tf.data.Dataset.
    Args:
        i (Integer): An integer with the index of the data in the Dataset instance.
        xy (Tuple): A Tuple of Tensor instances that contains the data and true label respectively.
    """
    return xy[0], xy[1]

def delete_patient_id(data, label, patient_id):
    """Function to delete the patient id variable return in a tf.data.Dataset.
    Args:
        data (Tensor): A Tensor instance that contains the video data.
        label (Tensor): A Tensor that contains the true label for the dataset.
        patient_id (Tensor): A Tensor that contains the patient id of the data.
    """
    return data, label

def move_frames_to_channels(data, label):
    """Function that move the frames axis to channels axis for 2D models and returns a tensor 
    with (height, width, frames) shape.
    Args:
        data (Tensor): A Tensor with 3 dimensions to be rearranged.
        label (Tensor): A Tensor that contains the true label for the dataset.
    """
    return tf.transpose(data, perm=[1,2,0]), label

def rgb_to_grayscale(data, label):
    """Function that change the data from RGB to Grayscale values in the last axis, also, removes the 
    dimension of 1 in gray data and returns a tensor with (frames, height, width) shape.
    Args:
        data (Tensor): A Tensor with 4 dimensions to be transposed being the last dimension of 3 (RGB).
        label (Tensor): A Tensor that contains the true label for the dataset.
    """
    gray_data = tf.image.rgb_to_grayscale(data)
    return tf.squeeze(gray_data), label