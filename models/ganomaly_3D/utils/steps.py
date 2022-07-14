"""This file contains the different steps (training and testing) for GANomaly 3D model.
Version: 1.2
Made by: Edgar Rangel
"""

import tensorflow as tf
from .losses import l1_loss, l2_loss, BCELoss

@tf.function
def train_step(gen_model, gen_opt, disc_model, disc_opt, x_data, w_gen = (1, 50, 1)):
    """Function that make one train step for whole GANomaly 3D model and returns the errors and relevant output variables.
    Args:
        gen_model (tf.keras.Model): An instance of the model generator to be trained.
        gen_opt (tf.keras.optimizers.Optimizer): An instance of keras optimizer element to be used as the generator optimizer to apply backpropagation with the gradients.
        disc_model (tf.keras.Model): An instance of the model discriminator to be trained.
        disc_opt (tf.keras.optimizers.Optimizer): An instance of keras optimizer element to be used as the discriminator optimizer to apply backpropagation with the gradients.
        x_data (Tensor Instance): A Tensor with the batched data to be given for the model in the step.
        w_gen (Tuple): An instance of tuple with 3 elements in the following order (w_adv, w_con, w_enc) to use in the error of generator.
    """
    assert len(w_gen) == 3
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #Forward process of networks
        fake, latent_i, latent_o = gen_model(x_data, training=True)
        pred_real, feat_real = disc_model(x_data, training=True)
        pred_fake, feat_fake = disc_model(fake, training=True)
        #Losses compute for generator
        err_g_adv = l2_loss(feat_fake, feat_real)
        err_g_con = l1_loss(x_data, fake)
        err_g_enc = l2_loss(latent_i, latent_o)
        err_g = err_g_adv * w_gen[0] + err_g_con * w_gen[1] +  err_g_enc * w_gen[2]
        #Losses compute for discriminator
        err_d_real = BCELoss(tf.ones_like(pred_real), pred_real)
        err_d_fake = BCELoss(tf.zeros_like(pred_fake), pred_fake)
        err_d = (err_d_real + err_d_fake) * 0.5

    gradients_g = gen_tape.gradient(err_g, gen_model.trainable_variables)
    gradients_d = disc_tape.gradient(err_d, disc_model.trainable_variables)

    gen_opt.apply_gradients(zip(gradients_g, gen_model.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_d, disc_model.trainable_variables))

    return err_g, err_d, fake, latent_i, latent_o, feat_real, feat_fake

@tf.function
def test_step(gen_model, disc_model, x_data):
    """Function that make one inference or eval step for whole GANomaly 3D model and returns its outputs to evaluate them.
    Args:
        gen_model (tf.keras.Model): An instance of the model generator to be trained.
        disc_model (tf.keras.Model): An instance of the model discriminator to be trained.
        x_data (Tensor Instance): A Tensor with the batched data to be given for the model in the step.
    """
    fake, latent_i, latent_o = gen_model(x_data, training=False)
    pred_real, feat_real = disc_model(x_data, training=False)
    pred_fake, feat_fake = disc_model(fake, training=False)
    return fake, latent_i, latent_o, feat_real, feat_fake