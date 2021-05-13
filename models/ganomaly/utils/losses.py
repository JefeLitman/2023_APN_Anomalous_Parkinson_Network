"""This file contains the losses for GANomaly nets translated in Tensorflow.
https://arxiv.org/abs/1805.06725
Version: 1.1
Made by: Edgar Rangel
"""

import tensorflow as tf

MAE = tf.keras.losses.MeanAbsoluteError()
MSE = tf.keras.losses.MeanSquaredError()
BCE = tf.keras.losses.BinaryCrossentropy()

def l1_loss(y_true, y_pred):
    """Function that return a function callback of the l1_loss used in GANomaly model.
    Args:
        y_true (Tensor): A Tensor with the real data of the dataset.
        y_pred (Tensor): A Tensor being the output of the model.
    """
    return MAE(y_true, y_pred)

def l2_loss(y_true, y_pred):
    """Function that return a function callback of the l2_loss used in GANomaly model.
    Args:
        y_true (Tensor): A Tensor with the real data of the dataset.
        y_pred (Tensor): A Tensor being the output of the model.
    """
    return MSE(y_true, y_pred)

def BCELoss(y_true, y_pred):
    """Function that return a function callback of the bce_loss used in GANomaly model.
    Args:
        y_true (Tensor): A Tensor with the real data of the dataset.
        y_pred (Tensor): A Tensor being the output of the model.
    """
    return BCE(y_true, y_pred)

def l1_loss_batch(y_true, y_pred):
    """Function that return the l1_loss sample by sample (no batch dimension reduction)
    used in GANomaly model.
    Args:
        y_true (Tensor): A Tensor with the real data of the dataset.
        y_pred (Tensor): A Tensor being the output of the model.
    """
    return tf.reduce_mean(tf.math.abs(y_true-y_pred), axis=range(1, tf.rank(y_true).numpy()))

def l2_loss_batch(y_true, y_pred):
    """Function that return the l2_loss sample by sample (no batch dimension reduction)
    used in GANomaly model.
    Args:
        y_true (Tensor): A Tensor with the real data of the dataset.
        y_pred (Tensor): A Tensor being the output of the model.
    """
    return tf.reduce_mean(tf.math.pow(y_true-y_pred, 2), axis=range(1, tf.rank(y_true).numpy()))
