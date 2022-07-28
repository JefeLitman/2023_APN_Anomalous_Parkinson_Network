"""This file contains the method to obtain GANomaly model.
https://arxiv.org/abs/1805.06725
Version: 1.1
Made by: Edgar Rangel
"""

import tensorflow as tf
from .networks.generator import get_generator
from .networks.discriminator import get_discriminator

def get_model(isize, nz, nc, ngf, extra_layers):
    """Function that return the GANomaly 3D model in tensorflow as a tuple (generator, discriminator) of instances models objects.
    Args:
        isize (Integer): Input size of the models.
        nz (Integer): Context vector size.
        nc (Integer): Quantity of channels of models input.
        ngf (Integer): Quantity of initial filters in the first convolution of the encoder.
        extra_layers (Integer): Quantity of layer blocks to add before reduction.
    Returns:
        generator_model: A tf.keras.Model instance.
        discriminator_model: A tf.keras.Model instance.
    """
    input_size = [isize]*3 + [nc]

    input_generator = tf.keras.Input(shape=input_size, name="Input-generator")
    gen_imag, latent_i, latent_o = get_generator(input_generator, isize, nz, nc, ngf, extra_layers)
    generator_model = tf.keras.Model(inputs=[input_generator], outputs=[gen_imag, latent_i, latent_o], name="Generator_3D")

    input_discriminator = tf.keras.Input(shape=input_size, name="Input-discriminator")
    classification, features = get_discriminator(input_discriminator, isize, nc, ngf, extra_layers)
    discriminator_model = tf.keras.Model(inputs=[input_discriminator], outputs=[classification, features], name="Discriminator_3D")

    return generator_model, discriminator_model