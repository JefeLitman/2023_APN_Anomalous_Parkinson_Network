"""This file contains methods to save videos, latent vectors and errors for all models.
Version: 1.3
Made by: Edgar Rangel
"""

import os
import cv2
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from .common import format_index, get_train_test_paths

def save_frame(frame, frame_index, folder_path):
    """Function that save a frame in the folder path given.
    Args:
        frame (Array): a 3D array with (h, w, c) shape where: h - height, w - width and c - channels with the video to be saved.
        frame_index (Integer): Number indicating the sequence of the frame that will be saved.
        folder_path (String): The root path where the video will be saved.
    """
    assert frame.ndim == 3
    if frame.shape[-1] == 1:
        convert = None
    elif frame.shape[-1] == 3:
        convert = cv2.COLOR_RGB2BGR
    else:
        raise AssertionError("The frame is not in RGB or gray scale to be saved.")

    if convert:
        frame = cv2.cvtColor(frame, convert)
    filename = format_index(frame_index) + ".png"
    cv2.imwrite(os.path.join(folder_path, filename), frame)

def save_video(video, folder_path):
    """Function that save a video frame by frame in the folder path given.
    Args:
        video (Array): a 4D array with (f, h, w, c) shape where f - frames, h - height, w - width and c - channels with the video to be saved.
        folder_path (String): The root path where the video will be saved.
    """
    assert video.ndim == 4

    for i, frame in enumerate(video):
        save_frame(frame, i + 1, folder_path)

def save_latent_vector(embedding_vector, folder_path, filename):
    """Function to save the given embedded vector as npy in the desired folder with an specific name. It doesn't return anything.
    Args:
        embedding_vector (Array): A numpy array with one dimension and shape (z) containing its elements to be saved.
        folder_path (String): The root path where the latent vector will be saved.
        filename (String): The name that will have the element to be saved.
    """
    assert embedding_vector.ndim == 1
    np.save(os.path.join(folder_path, filename), embedding_vector)

def save_errors(batch_errors, batch_labels, folder_path, training):
    """Function to save the batch of errors in the given folder path for train or test data, subdividing the normal and abnormal samples on different files.
    Args:
        batch_errors (Array): A 1D array with 1 dimension (shape of [batch]) with the errors to be saved.
        batch_labels (Array): A 1D array with shape with the labels of videos.
        folder_path (String): The root path where the errors will be saved.
        training (Boolean): Select True if the videos comes from the train data or False otherwise.
    """
    assert batch_labels.shape[0] == batch_errors.shape[0]

    normal_path, abnormal_path = get_train_test_paths(folder_path, training)

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
    ax = plt.gca()
    plt.plot(ax.get_xlim(), ax.get_ylim(), color="r")
    plt.title("QQ Plot for {}".format(filename))
    plt.savefig(os.path.join(save_path, filename+extension), dpi=300)
    plt.close()