"""This file contains the different methods to get the encoding dictionaries
for tfrecords datasets used in the notebooks.
Version: 1.0
Made by: Edgar Rangel
"""
import tensorflow as tf

def get_parkinson():
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
