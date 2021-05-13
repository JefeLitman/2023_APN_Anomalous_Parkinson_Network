"""This file contains methods to save the errors for GANomaly models.
Version: 1.0
Made by: Edgar Rangel
"""

import os
import numpy as np

def __make_subfolders__(folder_path, training):
    """Function that create the respectively subfolders to save the train/test normal 
    and abnormal videos or vectors and returns the normal and abnormal folder paths.
    Args:
        folder_path (String): The root path where the videos will be saved.
        training (Boolean): Select True if the videos comes from the train data or False otherwise.
    """
    if training:
        root_path = os.path.join(folder_path, "train")
    else:
        root_path = os.path.join(folder_path, "test")

    normal_path = os.path.join(root_path, "normal")
    abnormal_path = os.path.join(root_path, "abnormal")
        
    if not os.path.isdir(root_path):
        os.mkdir(root_path)
    if not os.path.isdir(normal_path):
        os.mkdir(normal_path)
    if not os.path.isdir(abnormal_path):
        os.mkdir(abnormal_path)

    return normal_path, abnormal_path

def save_errors(batch_errors, batch_labels, folder_path, training):
    """Function to save the batch of videos in the given folder path for train or test data, 
    subdividing the normal and abnormal samples on different folders.
    Args:
        batch_videos (Array): A 1D array with (b) shape with the errors to be saved.
        batch_labels (Array): A 1D array with (b) shape with the labels of videos.
        folder_path (String): The root path where the errors will be saved.
        training (Boolean): Select True if the videos comes from the train data or False otherwise.
    """
    assert batch_labels.shape[0] == batch_errors.shape[0]

    normal_path, abnormal_path = __make_subfolders__(folder_path, training)

    if os.path.isfile(normal_path+".npy"):
        normal_errors = np.load(normal_path+".npy")
    else:
        normal_errors = np.r_[[]]

    if os.path.isfile(abnormal_path+".npy"):
        abnormal_errors = np.load(abnormal_path+".npy")
    else:
        abnormal_errors = np.r_[[]]

    for i, label in enumerate(batch_labels):
        if label == 0:
            normal_errors = np.concatenate([normal_errors, [batch_errors[i]]])
        elif label == 1:
            abnormal_errors = np.concatenate([abnormal_errors, [batch_errors[i]]])
        else:
            raise AssertionError('There is an unknow label in the data. Label found {}, expected to be 0 or 1'.format(batch_labels[i]))

    if normal_errors.shape[0] != 0:
        np.save(normal_path, normal_errors)

    if abnormal_errors.shape[0] != 0:
        np.save(abnormal_path, abnormal_errors)