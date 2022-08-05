"""This file contains the loop for eval mode for GANomaly 3D model.
Version: 1.1
Made by: Edgar Rangel
"""

import os
import gc
import random
import numpy as np
import tensorflow as tf
from ..utils.steps import test_step
from ..utils.printers import get_metrics
from ..utils.weights_init import reinit_model
from ..utils.savers import save_models, save_model_results
from ..utils.exp_docs import experiment_folder_path, get_metrics_path, get_outputs_path, save_readme

def exec_loop(opts, TP, TN, FP, FN, AUC, train_data, val_data, test_data):
    """This function execute the loop for testing (a.k.a evaluation) in each epoch for GANomaly 3D model. It doesn't return anything but it will be showing the results obtained in each epoch.
    Args:
        opts (Dict): Dictionary that contains all the hiperparameters for the model, generally is the import of hiperparameters.py file of the model.
        TP (tf.keras.metrics): An instance of tf.keras.metrics.TruePositives which will work to calculate basic metrics.
        TN (tf.keras.metrics): An instance of tf.keras.metrics.TrueNegatives which will work to calculate basic metrics.
        FP (tf.keras.metrics): An instance of tf.keras.metrics.FalsePositives which will work to calculate basic metrics.
        FN (tf.keras.metrics): An instance of tf.keras.metrics.FalseNegatives which will work to calculate basic metrics.
        AUC (tf.keras.metrics): An instance of tf.keras.metrics.AUC which will work to calculate basic metrics.
        train_data (tf.data.Dataset): An instance of tf.data.Dataset containing the train data for the model.
        val_data (tf.data.Dataset): An instance of tf.data.Dataset containing the validation data for the model.
        test_data (tf.data.Dataset): An instance of tf.data.Dataset containing the test data for the model.
    """
    random.seed(opts["seed"])
    np.random.seed(opts["seed"])
    tf.random.set_seed(opts["seed"])

    experiment_path, _ = experiment_folder_path(opts["save_path"], opts["isize"], opts["nc"])
    metric_save_path = get_metrics_path(experiment_path)
    outputs_path = get_outputs_path(experiment_path)

    TP.reset_states()
    TN.reset_states()
    FP.reset_states()
    FN.reset_states()
    AUC.reset_states()

    gen_model = tf.keras.models.load_model(opts["gen_model_path"])
    disc_model = tf.keras.models.load_model(opts["disc_model_path"])

    if opts["eval_train"]:
        for step, xyi in enumerate(train_data):
            fake_images, latent_i, latent_o, feat_real, feat_fake = test_step(gen_model, disc_model, xyi[0])

            acc, pre, rec, spe, f1, auc = get_metrics(0, step, metric_save_path, xyi, opts['normal_class'], latent_i, latent_o, TP, TN, FP, FN, AUC, "train")
            save_model_results(xyi, fake_images, latent_i, latent_o, feat_real, feat_fake, outputs_path, "train", step == 0)

        TP.reset_states()
        TN.reset_states()
        FP.reset_states()
        FN.reset_states()
        AUC.reset_states()

    for step, xyi in enumerate(val_data):
        fake_images, latent_i, latent_o, feat_real, feat_fake = test_step(gen_model, disc_model, xyi[0])

        acc, pre, rec, spe, f1, auc = get_metrics(0, step, metric_save_path, xyi, opts['normal_class'], latent_i, latent_o, TP, TN, FP, FN, AUC, "val")
        save_model_results(xyi, fake_images, latent_i, latent_o, feat_real, feat_fake, outputs_path, "val", step == 0)

    TP.reset_states()
    TN.reset_states()
    FP.reset_states()
    FN.reset_states()
    AUC.reset_states()

    for step, xyi in enumerate(test_data):
        fake_images, latent_i, latent_o, feat_real, feat_fake = test_step(gen_model, disc_model, xyi[0])

        acc, pre, rec, spe, f1, auc = get_metrics(0, step, metric_save_path, xyi, opts['normal_class'], latent_i, latent_o, TP, TN, FP, FN, AUC, "test")
        save_model_results(xyi, fake_images, latent_i, latent_o, feat_real, feat_fake, outputs_path, "test", step == 0)
    
    ######################### Deleting the used model ###############################
    del gen_model
    del disc_model
    del train_data
    del val_data
    del test_data
    del xyi
    del fake_images
    del latent_i
    del latent_o
    del feat_real
    del feat_fake
    tf.keras.backend.clear_session()
    gc.collect()