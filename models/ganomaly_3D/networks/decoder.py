"""This file contains the decoder network used in GANomaly 3D net in Tensorflow.
Version: 1.1
Made by: Edgar Rangel
"""

import tensorflow as tf

from ..utils.weights_init import get_kernel_conv_init as conv_kernel_init
from ..utils.weights_init import get_gamma_batchNorm_init as batch_gamma_init
from ..utils.weights_init import get_beta_batchNorm_init as batch_beta_init

def get_decoder(x, isize, nz, nc, ngf, n_extra_layers=0):
    """Function that return the layers of Decoder for GANomaly 3D model in tensorflow.
    Args:
        x: Previous layer to be connected with this layer (Keras Layer Instance).
        isize: Input size of the previous layer (Integer).
        nz: Context vector size (Integer).
        nc: Quantity of channels to output (Integer).
        ngf: Quantity of initial filters in the first convolution of the encoder (Integer).
        n_extra_layers: Quantity of layer blocks to add before reduction (Integer).
    """
    if isize % 16 != 0:
        raise ValueError("isize has to be a multiple of 16")

    cngf, tisize = ngf // 2, 4
    while tisize != isize:
        cngf = cngf * 2
        tisize = tisize * 2

    x_prime = tf.keras.layers.Conv3DTranspose(filters=cngf, kernel_size=4, strides=1, 
        padding="valid", use_bias=False, kernel_initializer=conv_kernel_init(),
        name='initial-{0}-{1}-convt'.format(nz, cngf))(x)
    x_prime = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, 
        epsilon=1e-5, gamma_initializer=batch_gamma_init(), beta_initializer=batch_beta_init(),
        name='initial-{0}-batchnorm'.format(cngf))(x_prime)
    x_prime = tf.keras.layers.ReLU(name='initial-{0}-relu'.format(cngf))(x_prime)

    csize, _ = 4, cngf
    while csize < isize // 2:
        x_prime = tf.keras.layers.Conv3DTranspose(filters=cngf//2, kernel_size=4, strides=2, 
            padding="same", use_bias=False, kernel_initializer=conv_kernel_init(),
            name='pyramid-{0}-{1}-convt'.format(cngf, cngf // 2))(x_prime)
        x_prime = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, 
            epsilon=1e-5, gamma_initializer=batch_gamma_init(), beta_initializer=batch_beta_init(),
            name='pyramid-{0}-batchnorm'.format(cngf // 2))(x_prime)
        x_prime = tf.keras.layers.ReLU(name='pyramid-{0}-relu'.format(cngf // 2))(x_prime)
        cngf = cngf // 2
        csize = csize * 2

    # Extra layers
    for t in range(n_extra_layers):
        x_prime = tf.keras.layers.Conv3DTranspose(filters=cngf, kernel_size=3, strides=1, 
            padding="same", use_bias=False, kernel_initializer=conv_kernel_init(),
            name='extra-layers-{0}-{1}-conv'.format(t, cngf))(x_prime)
        x_prime = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.1, 
            epsilon=1e-5, gamma_initializer=batch_gamma_init(), beta_initializer=batch_beta_init(),
            name='extra-layers-{0}-{1}-batchnorm'.format(t, cngf))(x_prime)
        x_prime = tf.keras.layers.ReLU(name='extra-layers-{0}-{1}-relu'.format(t, cngf))(x_prime)

    x_prime = tf.keras.layers.Conv3DTranspose(filters=nc, kernel_size=4, strides=2, 
        padding="same", use_bias=False, kernel_initializer=conv_kernel_init(),
        name='final-{0}-{1}-convt'.format(cngf, nc))(x_prime)
    x_prime = tf.keras.layers.Activation(activation="tanh", name='final-{0}-tanh'.format(nc))(x_prime)
    
    return x_prime