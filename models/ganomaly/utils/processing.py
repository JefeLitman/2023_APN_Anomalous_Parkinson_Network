"""This file contains the different methods to process the data for GANomaly in Tensorflow.
Version: 1.5
Made by: Edgar Rangel
"""

import tensorflow as tf 

def normalize_accros_channels(data, label, mean, std, patient_id):
    """Function that normalize the tensors in the last axis (channels) with mean and standard deviation given.
    Args:
        data (Tensor): A Tensor with to be normalized in the last axis.
        label (Tensor): A Tensor that contains the true label for the dataset.
        mean (Decimal): An integer specifying the mean to normalize the data.
        std (Decimal): An integer specifying the standard deviation to normalize the data.
        patient_id (Tensor): A Tensor that contains the patient id of the data.
    """
    normalized = data - mean / std
    return normalized, label, patient_id

def min_max_scaler(data, label, new_min, new_max, patient_id):
    """Function that scale tensors using min max scaler formula letting the values between the new min and max values given.
    Args:
        data (Tensor): A Tensor with to be scaled in the last axis.
        label (Tensor): A Tensor that contains the true label for the dataset.
        new_min (Decimal): An integer specifying the new minimum value for the data.
        new_max (Decimal): An integer specifying the new maximum value for the data.
        patient_id (Tensor): A Tensor that contains the patient id of the data.
    """
    data_std = (data - tf.reduce_min(data)) / (tf.reduce_max(data) - tf.reduce_min(data))
    data_scaled = data_std * (new_max - new_min) + new_min
    return data_scaled, label, patient_id

def resize(data, label, size, patient_id):
    """Function that resize a 3D tensor frame by frame or 2D tensor given a specified size.
    Args:
        data (Tensor): A Tensor with 4 or 3 dimension to be resized.
        label (Tensor): A Tensor that contains the true label for the dataset.
        size (Tuple): A Tuple with 2 elements (height, width) respectively being the new size of the frames.
        patient_id (Tensor): A Tensor that contains the patient id of the data.
    """
    return tf.image.resize(data, size), label, patient_id

def get_center_of_volume(data, label, n_frames, patient_id):
    """Function that get the center portion of 3D tensor (video) given the n frames.
    Args:
        data (Tensor): A Tensor with 4 dimension to be cropped with (frames, height, width, channels) shape.
        label (Tensor): A Tensor that contains the true label for the dataset.
        n_frames (Integer): An integer specifying the quantity of frames to extract from data.
        patient_id (Tensor): A Tensor that contains the patient id of the data.
    """
    half = tf.math.floordiv(tf.shape(data)[0], 2)
    return data[half - n_frames//2 : half + n_frames//2], label, patient_id

def oversampling_equidistant_full(data, label, n_frames, patient_id):
    """Function that make always an oversampling of the data (video) with at least one extended sample or multiple samples extended if the data have more frames than the required.
    Args:
        data (Tensor): A Tensor with 4 dimension to be cropped with (frames, height, width, channels) shape.
        label (Tensor): A Tensor that contains the true label for the dataset.
        n_frames (Integer): An integer specifying the quantity of frames to extract from data.
        patient_id (Tensor): A Tensor that contains the patient id of the data.
    """
    # Calculating basic parameters
    x = data
    x_frames = tf.shape(x)[0]
    q = tf.cast(tf.math.ceil(x_frames / n_frames), tf.int32)
    k = tf.abs(q * n_frames - x_frames)
    
    #Condition to repeat the center if its possible
    if k <= x_frames:
        half_x = tf.cast(tf.math.floor(x_frames / 2), tf.int32)
        half_k = tf.cast(tf.math.floor(k / 2), tf.int32)
        if k % 2 == 0:
            augmented_volume = tf.repeat(x[half_x - half_k : half_x + half_k], [2], axis=0)
            total_volume = tf.concat([x[:half_x - half_k], augmented_volume, x[half_x + half_k:]], axis=0)
        else:
            augmented_volume = tf.repeat(x[half_x - half_k : half_x + half_k + 1], [2], axis=0)
            total_volume = tf.concat([x[:half_x - half_k], augmented_volume, x[half_x + half_k + 1:]], axis=0)
    else:
        k = tf.cast(tf.math.ceil(n_frames / x_frames), tf.int32)
        augmented_volume = tf.repeat(x, [k], axis=0)
        half_x = tf.cast(tf.math.floor((k * x_frames) / 2), tf.int32)
        half_k = tf.cast(tf.math.floor(n_frames / 2), tf.int32)
        total_volume = augmented_volume[half_x - half_k : half_x + half_k]
        
    volumes = total_volume[0::q]
    for i in tf.range(1, q):
        volumes = tf.concat([volumes, total_volume[i::q]], axis=0)
    samples = tf.reshape(volumes, tf.concat([[q, n_frames], tf.shape(x)[1:]], axis=0))
    
    labels = tf.tile([label], [q]) # Repeat the label 
    ids = tf.tile([patient_id], [q]) # Repeat the patient_id 
    xs = tf.data.Dataset.from_tensor_slices(samples)
    ys = tf.data.Dataset.from_tensor_slices(labels)
    zs = tf.data.Dataset.from_tensor_slices(ids)
    return tf.data.Dataset.zip((xs, ys, zs))

def undo_enumerate(i, xyi):
    """Function that revert the enumerate method applied over a tf.data.Dataset.
    Args:
        i (Integer): An integer with the index of the data in the Dataset instance.
        xyi (Tuple): A Tuple of Tensor instances that contains the data, true label and patiend id respectively.
    """
    return xyi[0], xyi[1], xyi[2]

def move_frames_to_channels(data, label, patient_id):
    """Function that move the frames axis to channels axis for 2D models and returns a tensor 
    with (height, width, frames) shape.
    Args:
        data (Tensor): A Tensor with 3 dimensions to be rearranged.
        label (Tensor): A Tensor that contains the true label for the dataset.
        patient_id (Tensor): A Tensor that contains the patient id of the data.
    """
    return tf.transpose(data, perm=[1,2,0]), label, patient_id

def rgb_to_grayscale(data, label, reduce_gray_dimension, patient_id):
    """Function that change the data from RGB to Grayscale values in the last axis, also, removes the 
    dimension of 1 in gray data and returns a tensor with (frames, height, width) shape.
    Args:
        data (Tensor): A Tensor with 4 dimensions to be transposed being the last dimension of 3 (RGB).
        label (Tensor): A Tensor that contains the true label for the dataset.
        reduce_gray_dimension (Boolean): A Boolean specifying if the channels dimensions must be reduced or deleted.
        patient_id (Tensor): A Tensor that contains the patient id of the data.
    """
    gray_data = tf.image.rgb_to_grayscale(data)
    if reduce_gray_dimension:
        return tf.squeeze(gray_data), label, patient_id
    else:
        return gray_data, label, patient_id

def repeat_and_identify_frames(data, label, patient_id):
    """This function repeat the label and patient_id the quantity of frames data contains and
    returns them with a new attribute being the frame_id.
    Args:
        data (Tensor): A Tensor with 4 dimensions to be unbatched.
        label (Tensor): A Tensor that contains the true label for the dataset.
        patient_id (Tensor): A Tensor that contains the patient id of the data.
    """
    labels = tf.repeat(label, [tf.shape(data)[0]], axis=0)
    ids = tf.repeat(patient_id, [tf.shape(data)[0]], axis=0)
    frames_id = tf.range(1, tf.shape(data)[0] + 1, 1, tf.int64)
    return data, labels, ids, frames_id

def add_video_id(i, xyif):
    """Function that revert the enumerate method applied over a tf.data.Dataset but it adds the
    parameter of video id to the data.
    Args:
        i (Integer): An integer with the index of the data in the Dataset instance.
        xyif (Tuple): A Tuple of Tensor instances that contains the data, true label, patiend id 
        and frame id respectively.
    """
    video_id = tf.repeat(i + 1, [tf.shape(xyif[0])[0]], axis=0)
    return xyif[0], xyif[1], xyif[2], xyif[3], video_id