"""This file contains the whole GANomaly models translated to Tensorflow.
https://arxiv.org/abs/1805.06725
Version: 1.0
Made by: Edgar Rangel
"""

import tensorflow as tf

from .networks._2D.generator import get_generator as generator_2D
from .networks._2D.discriminator import get_discriminator as discriminator_2D
from .networks._3D.generator import get_generator as generator_3D
from .networks._3D.discriminator import get_discriminator as discriminator_3D

def get_2D_models(isize, nz, nc, ngf, extra_layers):
    """Function that return the GANomaly model translated 
    to tensorflow as a tuple (generator, discriminator) of 
    instances models objects.
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
    input_size = [isize]*2 + [nc]

    input_generator = tf.keras.Input(shape=input_size, name="Input-generator")
    gen_imag, latent_i, latent_o = generator_2D(input_generator, isize, nz, nc, ngf, extra_layers)
    generator_model = tf.keras.Model(inputs=[input_generator], outputs=[gen_imag, latent_i, latent_o], name="Generator_2D")

    input_discriminator = tf.keras.Input(shape=input_size, name="Input-discriminator")
    classification, features = discriminator_2D(input_discriminator, isize, nc, ngf, extra_layers)
    discriminator_model = tf.keras.Model(inputs=[input_discriminator], outputs=[classification, features], name="Discriminator_2D")

    return generator_model, discriminator_model

def get_3D_models(isize, nz, nc, ngf, extra_layers):
    """Function that return the GANomaly model translated to 3D
    in tensorflow as a tuple (generator, discriminator) of 
    instances models objects.
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
    gen_imag, latent_i, latent_o = generator_3D(input_generator, isize, nz, nc, ngf, extra_layers)
    generator_model = tf.keras.Model(inputs=[input_generator], outputs=[gen_imag, latent_i, latent_o], name="Generator_3D")

    input_discriminator = tf.keras.Input(shape=input_size, name="Input-discriminator")
    classification, features = discriminator_3D(input_discriminator, isize, nc, ngf, extra_layers)
    discriminator_model = tf.keras.Model(inputs=[input_discriminator], outputs=[classification, features], name="Discriminator_3D")

    return generator_model, discriminator_model