"""This file contains the losses for GANomaly nets translated in Tensorflow.
https://arxiv.org/abs/1805.06725
Version: 1.0
Made by: Edgar Rangel
"""

import tensorflow as tf

MAE = tf.keras.losses.MeanAbsoluteError()
MSE = tf.keras.losses.MeanSquaredError()
BCE = tf.keras.losses.BinaryCrossentropy()

def l1_loss(y_true, y_pred):
    """Function that return a function callback of the l1_loss used in GANomaly model.
    Args:
        y_true: A Tensor with the real data of the dataset (Tensor Instance).
        y_pred: A Tensor being the output of the model (Tensor Instance).
    """
    return MAE(y_true, y_pred)

def l2_loss(y_true, y_pred):
    """Function that return a function callback of the l2_loss used in GANomaly model.
    Args:
        y_true: A Tensor with the real data of the dataset (Tensor Instance).
        y_pred: A Tensor being the output of the model (Tensor Instance).
    """
    return MSE(y_true, y_pred)

def BCELoss(y_true, y_pred):
    """Function that return a function callback of the bce_loss used in GANomaly model.
    Args:
        y_true: A Tensor with the real data of the dataset (Tensor Instance).
        y_pred: A Tensor being the output of the model (Tensor Instance).
    """
    return BCE(y_true, y_pred)