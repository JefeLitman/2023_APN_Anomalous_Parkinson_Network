"""This file contains the generator network used in GANomaly net translated to 3D in Tensorflow.
https://arxiv.org/abs/1805.06725
Version: 1.0
Made by: Edgar Rangel
"""

from .encoder import get_encoder
from .decoder import get_decoder

def get_generator(x, isize, nz, nc, ngf, extra_layers):
    """Function that return the Generator model for GANomaly model translated 
    in tensorflow.
    Args:
        x: Previous layer to be connected with this layer (Keras Layer Instance).
        isize: Input size of the previous layer (Integer).
        nz: Context vector size (Integer).
        nc: Quantity of channels in input (Integer).
        ngf: Quantity of initial filters in the first convolution of the encoder (Integer).
        extra_layers: Quantity of layer blocks to add before reduction (Integer).
    """
    latent_i, _ = get_encoder(x, isize, nz, nc, ngf, extra_layers, 1)
    gen_imag = get_decoder(latent_i, isize, nz, nc, ngf, extra_layers)
    latent_o, _ = get_encoder(gen_imag, isize, nz, nc, ngf, extra_layers, 2)
    return gen_imag, latent_i, latent_o