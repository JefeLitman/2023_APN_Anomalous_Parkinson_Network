"""This file contains methods to save the errors for GANomaly models.
Version: 1.1.2
Made by: Edgar Rangel
"""

import os
import cv2
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

    return normal_path, abnormal_path

def __format_number__(index):
    """This function format the index integer into a string in the range of 0000 to 9999.
    Args:
        index (Int): Integer to be formatted.
    """
    if index < 10:
        return '000' + str(index)
    elif index < 100:
        return '00' + str(index)
    elif index < 1000:
        return '0' + str(index)
    else:
        return str(index)

def save_errors(batch_errors, batch_labels, folder_path, training):
    """Function to save the batch of errors in the given folder path for train or test data, 
    subdividing the normal and abnormal samples on different files.
    Args:
        batch_errors (Array): A 1D array with (b) shape with the errors to be saved.
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

def save_frames(batch_frames, batch_labels, batch_ids, batch_video_id, batch_frame_id, folder_path, training):
    """Function to save the batch of frames in the given folder path for train or test data, 
    subdividing the normal and abnormal samples on different folders and making a folder per 
    patient and saving its frames by order.
    Args:
        batch_frames (Array): A 4D array with (b, h, w, c) shapes respectively 
        where b - batch size, f - frames, h - height, w - width and 
        c - channels with the videos to be saved.
        batch_labels (Array): A 1D array with (b) shape with the labels of videos.
        batch_ids (Array): A 1D array with (b) shape with the patient id of every video.
        batch_video_id (Array): A 1D array with (b) shape indicating what video of the patient is saved.
        batch_frame_id (Array): A 1D array with (b) shape indicating what frame of the video is.
        folder_path (String): The root path where the videos will be saved.
        model_dimension (String): Dimensionality on which the model's convolutions operate, can be "3D" or "2D".
        training (Boolean): Select True if the videos comes from the train data or False otherwise.
    """
    assert batch_labels.shape[0] == batch_frames.shape[0]
            
    normal_path, abnormal_path = __make_subfolders__(folder_path, training)
    if not os.path.isdir(normal_path):
        os.mkdir(normal_path)
    if not os.path.isdir(abnormal_path):
        os.mkdir(abnormal_path)

    for i, frame in enumerate(batch_frames):
        if batch_labels[i] == 0:
            save_path =  normal_path
        elif batch_labels[i] == 1:
            save_path = abnormal_path
        else:
            raise AssertionError('There is an unknow label in the data. Label found {}, expected to be 0 or 1'.format(batch_labels[i]))

        folder = "patient-{}_video-{}".format(
            __format_number__(batch_ids[i]),
            __format_number__(batch_video_id[i])
        )
        save_path = os.path.join(save_path, folder)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        
        assert frame.ndim == 3
        if frame.shape[-1] == 1:
            convert = None
        elif frame.shape[-1] == 3:
            convert = cv2.COLOR_RGB2BGR
        else:
            raise AssertionError("The video is not in RGB or gray scale to be saved.")

        if convert:
            frame_converted = cv2.cvtColor(frame, convert)
        else:
            frame_converted = frame
        filename = __format_number__(batch_frame_id[i]) + ".png"
        cv2.imwrite(os.path.join(save_path, filename), frame_converted)