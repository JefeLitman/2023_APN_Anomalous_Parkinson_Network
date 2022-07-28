"""This file contains the discriminator network used in GANomaly net translated to Tensorflow.
https://arxiv.org/abs/1805.06725
Version: 1.1
Made by: Edgar Rangel
"""

from .encoder import get_encoder

def get_discriminator(x, isize, nc, ngf, extra_layers):
    """Function that return the Discriminator model for GANomaly model translated in tensorflow.
    Args:
        x (tf.keras.Layer): Previous layer to be connected with this layer.
        isize (Integer): Input size of the previous layer.
        nc (Integer): Quantity of channels in input.
        ngf (Integer): Quantity of initial filters in the first convolution of the encoder.
        extra_layers (Integer): Quantity of layer blocks to add before reduction.
    """
    
    return get_encoder(x, isize, 1, nc, ngf, extra_layers, 3)