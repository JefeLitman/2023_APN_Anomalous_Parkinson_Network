"""This file contains the generator network used in GANomaly net translated to Tensorflow.
https://arxiv.org/abs/1805.06725
Version: 1.1
Made by: Edgar Rangel
"""

from .encoder import get_encoder
from .decoder import get_decoder

def get_generator(x, isize, nz, nc, ngf, extra_layers):
    """Function that return the Generator model for GANomaly model translated in tensorflow.
    Args:
        x: Previous layer to be connected with this layer (Keras Layer Instance).
        isize (Integer): Input size of the previous layer.
        nz (Integer): Context vector size.
        nc (Integer): Quantity of channels in input.
        ngf (Integer): Quantity of initial filters in the first convolution of the encoder.
        extra_layers (Integer): Quantity of layer blocks to add before reduction.
    """
    latent_i, _ = get_encoder(x, isize, nz, nc, ngf, extra_layers, 1)
    gen_imag = get_decoder(latent_i, isize, nz, nc, ngf, extra_layers)
    latent_o, _ = get_encoder(gen_imag, isize, nz, nc, ngf, extra_layers, 2)
    return gen_imag, latent_i, latent_o