"""This file contains the encoder network used in GANomaly net translated to Tensorflow.
https://arxiv.org/abs/1805.06725
Version: 1.1
Made by: Edgar Rangel
"""

import tensorflow as tf

from ..utils.weights_init import get_kernel_conv_init as conv_kernel_init
from ..utils.weights_init import get_gamma_batchNorm_init as batch_gamma_init
from ..utils.weights_init import get_beta_batchNorm_init as batch_beta_init

def get_encoder(x, isize, nz, nc, ndf, n_extra_layers=0, encoder_id=0, add_final_conv=True):
    """Function that add and forward the layers of Encoder for GANomaly model translated in tensorflow. This method return two variables, the classifier output and embedding vector.
    Args:
        x (tf.keras.Layer): Previous layer to be connected with this layer.
        isize (Integer): Input size of the previous layer.
        nz (Integer): Context vector size.
        nc (Integer): Quantity of channels in input.
        ndf (Integer): Quantity of initial filters in the first convolution.
        n_extra_layers (Integer): Quantity of layer blocks to add before reduction.
        encoder_id (Integer): Number id to make encoder unique among all enconders.
        add_final_conv (Boolean): Add the final convolution with nz filters and unitary dimensions.
    """
    if isize % 16 != 0:
        raise ValueError("isize has to be a multiple of 16")

    x_prime = tf.keras.layers.Conv2D(filters=ndf, kernel_size=4, strides=2, 
            padding="same", use_bias=False, kernel_initializer=conv_kernel_init(),
            name='initial-conv-{0}-{1}-{2}'.format(nc, ndf, encoder_id))(x)
    x_prime = tf.keras.layers.LeakyReLU(alpha=0.2, name='initial-relu-{0}-{1}'.format(ndf, encoder_id))(x_prime)

    csize, cndf = isize / 2, ndf

    # Extra layers
    for t in range(n_extra_layers):
        x_prime = tf.keras.layers.Conv2D(filters=cndf, kernel_size=3, strides=1, 
            padding="same", use_bias=False, kernel_initializer=conv_kernel_init(),
            name='extra-layers-{0}-{1}-conv-{2}'.format(t, cndf, encoder_id))(x_prime)
        x_prime = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, 
            epsilon=1e-5, gamma_initializer=batch_gamma_init(), beta_initializer=batch_beta_init(),
            name='extra-layers-{0}-{1}-batchnorm-{2}'.format(t, cndf, encoder_id))(x_prime)
        x_prime = tf.keras.layers.LeakyReLU(alpha=0.2, 
            name='extra-layers-{0}-{1}-relu-{2}'.format(t, cndf, encoder_id))(x_prime)

    while csize > 4:
        in_feat = cndf
        out_feat = cndf * 2
        x_prime = tf.keras.layers.Conv2D(filters=out_feat, kernel_size=4, strides=2, 
            padding="same", use_bias=False, kernel_initializer=conv_kernel_init(),
            name='pyramid-{0}-{1}-conv-{2}'.format(in_feat, out_feat, encoder_id))(x_prime)
        x_prime = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, 
            epsilon=1e-5, gamma_initializer=batch_gamma_init(), beta_initializer=batch_beta_init(), 
            name='pyramid-{0}-batchnorm-{1}'.format(out_feat, encoder_id))(x_prime)
        x_prime = tf.keras.layers.LeakyReLU(alpha=0.2, 
            name='pyramid-{0}-relu-{1}'.format(out_feat, encoder_id))(x_prime)
        cndf = cndf * 2
        csize = csize / 2

    if add_final_conv:
        classification = tf.keras.layers.Conv2D(filters=nz, kernel_size=4, strides=1, 
            padding="valid", use_bias=False, kernel_initializer=conv_kernel_init(),
            name='final-{0}-{1}-conv-{2}'.format(cndf, 1, encoder_id))(x_prime)
    
    return classification, x_prime #x_prime as features in GANomaly
    