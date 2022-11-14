"""This file contains the basic extraction methods for the tfrecord hosted in this folder. The only thing to do is call the methods get_data to obtain the raw data from the tfrecord.
Version: 1.0
Made by: Edgar Rangel
"""

import tensorflow as tf

def __get_encoding_dict__():
    """This function returns the encoding dictionary for the tfrecord used in this instance for KOA_PD_NM dataset."""
    encoding_dictionary = {
        'label': tf.io.FixedLenFeature([], tf.int64), # 0 -> NM, 1 -> PD, 2 -> KOA
        'level': tf.io.FixedLenFeature([], tf.int64), # 0 -> EL;ML;NM, 1 -> MD, 2 -> SV
        'id': tf.io.FixedLenFeature([], tf.int64),  # Integer with the patient id
        'frames': tf.io.FixedLenFeature([], tf.int64), # Total numbers of frames in the video
        'height': tf.io.FixedLenFeature([], tf.int64), # Height of the frames
        'width': tf.io.FixedLenFeature([], tf.int64), # Width of the frames
        'channels': tf.io.FixedLenFeature([], tf.int64), # Channels of the frames, default in 3
        'video': tf.io.FixedLenFeature([], tf.string) # The video data itself
    }
    return encoding_dictionary

def __extract_data_from_dict__(example_dict):
    """This function take a dictionary with all the data stored in a tfrecord and return the desired elements from it. See the encoding dictionary to know what data is stored in the tfrecord."""
    f = example_dict["frames"]
    h = example_dict["height"]
    w = example_dict["width"]
    c = example_dict["channels"]
    raw_volume = tf.io.decode_raw(example_dict["video"], tf.uint8)
    volume = tf.reshape(raw_volume, [f, h, w, c])
    return tf.cast(volume, dtype=tf.float32), example_dict["label"], example_dict["level"], example_dict["id"]

def get_data(dataset_path, n_cpus):
    """This function return an instance of ParrallelMapDataset or TFRecordDataset to be called and used with a for while its passing through all the data.
    Args:
        dataset_path (String): Path in which the tfrecord is allocated.
        n_cpus (Int): Quantity of CPUs to be used while the data is being read it and processed.
    """
    encoding_dictionary = __get_encoding_dict__()
    raw_data = tf.data.TFRecordDataset(dataset_path)
    dict_data = raw_data.map(lambda x: tf.io.parse_single_example(x, encoding_dictionary))
    total_data = dict_data.map(__extract_data_from_dict__, n_cpus)
    return total_data