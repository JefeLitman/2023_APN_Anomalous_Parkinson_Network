"""This file contains common functions used across different metrics to use with the models.
Version: 1.1
Made by: Edgar Rangel
"""

import os

def format_index(index, max_digits = 4):
    """This function format the index integer into a string with the maximun quantity of digits given.
    Args:
        index (Int): Integer to be formatted.
        max_digits (Int): How many digits must the number contain, e.g: if 4 then the range is from 0000 to 9999.
    """
    value = str(index)
    while len(value) < max_digits:
        value = '0' + value
    return value

def get_partitions_paths(base_path, partition):
    """Function that returns two paths for normal or abnormal data that will be saved. It take into account if the data will be saved for train or test data and create the respective folder. The first returned folder is for normal and last for abnormal.
    Args:
        base_path (String): The root path where the data will be saved.
        partition (String): The partitions to create the folder, the available options are "train", "val" or "test".
    """
    if partition.lower() in ["train", "test", "val"]:
        root_path = os.path.join(base_path, partition.lower())
    else:
        raise ValueError('You give an unknow partition to create the folder to contain that data. The partition given was "{}"'.format(partition))

    normal_path = os.path.join(root_path, "normal")
    abnormal_path = os.path.join(root_path, "abnormal")
        
    if not os.path.isdir(root_path):
        os.mkdir(root_path)

    return normal_path, abnormal_path

def get_next_last_item(base_path):
    """Function than will return an integer with an id of the following element for the last item listed in the base path. Also, this method checks it the elements in the folder are formatted with the following order (xxxx_*) where xxxx is a sequences of integers.
    Args:
        base_path (String): The root path in where the elements will be listed.
    """
    elements = [int(i.split("_")[0]) for i in sorted(os.listdir(base_path))]
    if len(elements) == 0:
        return 1
    else:
        return elements[-1] + 1