"""This file contains the weight initializer for different layers in GANomaly 3D model.
Version: 1.1
Made by: Edgar Rangel
"""

import tensorflow as tf

def get_kernel_conv_init():
    """Function that return the initializer for convolutions kernels in GANomaly 3D model."""
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

def get_gamma_batchNorm_init():
    """Function that return the gamma initializer for BatchNormalization layers in GANomaly 3D model."""
    return tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02)

def get_beta_batchNorm_init():
    """Function that return the beta initializer for BatchNormalization layers in GANomaly 3D model."""
    return tf.keras.initializers.Zeros()

def reinit_model(model):
    """Function that reset the weights of the given model using the initializers in its layers.
    Args:
        model (Keras Model): An instance of tensorflow keras model to reinitialize its weights.
    """
    for layer in model.layers:
        nombre = layer.get_config()["name"]
        weights = []
        if "conv" in nombre:
            for i, weight in enumerate(layer.get_weights()):
                if i == 0:
                    weights.append(layer.kernel_initializer(weight.shape))
                else:
                    weights.append(layer.bias_initializer(weight.shape))
        elif "batchnorm" in nombre:
            for i, weight in enumerate(layer.get_weights()):
                if i == 0:
                    weights.append(layer.gamma_initializer(weight.shape))
                elif i == 1:
                    weights.append(layer.beta_initializer(weight.shape))
                elif i == 2:
                    weights.append(layer.moving_mean_initializer(weight.shape))
                else:
                    weights.append(layer.moving_variance_initializer(weight.shape))
        layer.set_weights(weights)
