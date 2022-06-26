"""This file contains the discriminator network used in GANomaly 3D net in Tensorflow.
Version: 1.1
Made by: Edgar Rangel
"""

from .encoder import get_encoder

def get_discriminator(x, isize, nc, ngf, extra_layers):
    """Function that return the Discriminator model for GANomaly 3D model in tensorflow.
    Args:
        x: Previous layer to be connected with this layer (Keras Layer Instance).
        isize: Input size of the previous layer (Integer).
        nc: Quantity of channels in input (Integer).
        ngf: Quantity of initial filters in the first convolution of the encoder (Integer).
        extra_layers: Quantity of layer blocks to add before reduction (Integer).
    """
    
    return get_encoder(x, isize, 1, nc, ngf, extra_layers, 3)