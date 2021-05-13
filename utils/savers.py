"""This file contains methods to save videos and latent vectors for all models.
Version: 1.1.1
Made by: Edgar Rangel
"""

import os
import cv2
import matplotlib.pyplot as plt
from numpy.lib.npyio import save
import statsmodels.api as sm
import numpy as np

def __save_video__(video, folder_path):
    """Function that save a video frame by frame in the folder path given.
    Args:
        video (Array): a 4D array with (f, h, w, c) shape where f - frames, 
        h - height, w - width and c - channels with the video to be saved.
        folder_path (String): The root path where the video will be saved.
    """
    assert video.ndim == 4
    if video.shape[-1] == 1:
        convert = None
    elif video.shape[-1] == 3:
        convert = cv2.COLOR_RGB2BGR
    else:
        raise AssertionError("The video is not in RGB or gray scale to be saved.")

    for i, frame in enumerate(video):
        if convert:
            frame = cv2.cvtColor(frame, convert)

        if i + 1 < 10:
            filename = '000' + str(i + 1) + ".png"
        elif i + 1 < 100:
            filename = '00' + str(i + 1) + ".png"
        elif i + 1 < 1000:
            filename = '0' + str(i + 1) + ".png"
        else:
            filename = str(i + 1) + ".png"

        cv2.imwrite(os.path.join(folder_path, filename), frame)

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

def __get_last_item__(folder_path):
    """Function that return the next item name of elements in the folder starting
    with 0001. The folder must have folders named like 0001, 0002, 0010, ...
    Args:
        folder_path (String): The path where will be checked its items.
    """
    items = [int(i) for i in sorted(os.listdir(folder_path))]
    index = len(items)
    if index + 1 < 10:
        item = '000' + str(index + 1)
    elif index + 1 < 100:
        item = '00' + str(index + 1)
    elif index + 1 < 1000:
        item = '0' + str(index + 1)
    else:
        item = str(index + 1)
    return item

def save_videos(batch_videos, batch_labels, folder_path, model_dimension, training):
    """Function to save the batch of videos in the given folder path for train or test data, 
    subdividing the normal and abnormal samples on different folders.
    Args:
        batch_videos (Array): A 4D or 5D array with (b, h, w, f) or (b, f, h, w, c) 
        shapes respectively where b - batch size, f - frames, h - height, w - width and 
        c - channels with the videos to be saved.
        batch_lables (Array): A 1D array with (b) shape with the labels of videos.
        folder_path (String): The root path where the videos will be saved.
        model_dimension (String): Dimensionality on which the model's convolutions operate, can be "3D" or "2D".
        training (Boolean): Select True if the videos comes from the train data or False otherwise.
    """
    assert batch_labels.shape[0] == batch_videos.shape[0]
    assert model_dimension in ["2D", "3D"]
    if model_dimension == "2D":
        if batch_videos.ndim == 4:
            batch_videos = np.expand_dims(np.moveaxis(batch_videos, 3, 1), -1)
        else:
            raise AssertionError('The quantity of dimension for batch_videos must be 4 in model dimension "2D".')
    elif model_dimension == "3D":
        if batch_videos.ndim == 5:
            pass
        else:
            raise AssertionError('The quantity of dimension for batch_videos must be 5 in model dimension "3D".')
            
    normal_path, abnormal_path = __make_subfolders__(folder_path, training)

    for i, video in enumerate(batch_videos):
        if batch_labels[i] == 0:
            save_path =  normal_path
        elif batch_labels[i] == 1:
            save_path = abnormal_path
        else:
            raise AssertionError('There is an unknow label in the data. Label found {}, expected to be 0 or 1'.format(batch_labels[i]))

        folder = __get_last_item__(save_path)
        save_path = os.path.join(save_path, folder)
        os.mkdir(save_path)
        __save_video__(video, save_path)

def save_latent_vectors(batch_latent, batch_labels, folder_path, training):
    """Function to save the latent vectors of videos in the given folder path for train or test data, 
    subdividing the normal and abnormal samples on different folders.
    Args:
        batch_latent (Array): A 2D with (b, z) shape where b - batch size, z - context vector size
        with the latent_vectors to be saved.
        batch_lables (Array): A 1D array with (b) shape with the labels of videos.
        folder_path (String): The root path where the latent vectors will be saved.
        training (Boolean): Select True if the videos comes from the train data or False otherwise.
    """
    assert batch_labels.shape[0] == batch_latent.shape[0]
    assert batch_latent.ndim == 2

    normal_path, abnormal_path = __make_subfolders__(folder_path, training)

    for i, vector in enumerate(batch_latent):
        if batch_labels[i] == 0:
            save_path =  normal_path
        elif batch_labels[i] == 1:
            save_path = abnormal_path
        else:
            raise AssertionError('There is an unknow label in the data. Label found {}, expected to be 0 or 1'.format(batch_labels[i]))

        filename = __get_last_item__(save_path)
        np.save(os.path.join(save_path, filename), vector)

def generate_qq_plot(data, save_path, filename, extension=".png", normal_mean = 0, normal_std = 1):
    """Function that generate and save a qq-plot to visualize if the data follows a normal distribution.
    Args:
        data (Array): An 1D array of data containing the values to be compared against a normal pdf.
        save_path (String): The path where the figure will be saved.
        filename (String): Name of the plot to be saved.
        extension (String): Extension of the image to be saved, default in .png.
        normal_mean (Decimal): The value of the mean for the normal distribution.
        normal_std (Decimal): The value of the standard deviation for the normal distribution.
    """
    sm.qqplot(data, loc = normal_mean, scale = normal_std)
    plt.title("QQ Plot for {}".format(filename))
    plt.savefig(os.path.join(save_path, filename+extension))
    plt.close()