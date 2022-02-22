"""This file contains the different get methods for the encoding dictionaries
of tfrecords datasets used. The method must be named "get_<model>" e.g. get_ganomaly
Version: 1.1
Made by: Edgar Rangel
"""
import tensorflow as tf

def get_ganomaly():
    """Function to get the encoding dictionary for Parkinson cutted frames and 
    total frames dataset."""
    encoding_dictionary = {
        'parkinson': tf.io.FixedLenFeature([], tf.int64), # 0 -> Normal, 1 -> Parkinson
        'id': tf.io.FixedLenFeature([], tf.int64), # Integer with the patient id
        'frames': tf.io.FixedLenFeature([], tf.int64), # Total numbers of frames in the video
        'height': tf.io.FixedLenFeature([], tf.int64), # Height of the frames
        'width': tf.io.FixedLenFeature([], tf.int64), # Width of the frames
        'channels': tf.io.FixedLenFeature([], tf.int64), # Channels of the frames, default in 3
        'video': tf.io.FixedLenFeature([], tf.string), # The video data itself
    }
    return encoding_dictionary
