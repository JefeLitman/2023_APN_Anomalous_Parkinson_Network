"""This file contains the loop for train and eval mode for GANomaly 3D model.
Version: 1.1
Made by: Edgar Rangel
"""

import os
import gc
import random
import numpy as np
import tensorflow as tf
from ..model import get_model
from ..utils.printers import get_metrics
from ..utils.weights_init import reinit_model
from ..utils.steps import train_step, test_step
from ..utils.savers import save_models, save_model_results
from ..utils.exp_docs import experiment_folder_path, get_metrics_path, get_outputs_path, save_readme

def exec_loop(opts, readme_template, kfold, TP, TN, FP, FN, AUC, gen_loss, disc_loss, train_data, test_data, normal_class):
    """This function execute the loop for training and evaluation in each epoch for GANomaly 3D model. It doesn't return anything but it will be showing the results obtained in each epoch.
    Args:
        opts (Dict): Dictionary that contains all the hiperparameters for the model, generally is the import of hiperparameters.py file of the model.
        readme_template (String): A string containing the help text which will be saved along the experiment elements.
        kfold (Integer): An integer indicating in which kfold the loop is executed.
        TP (tf.keras.metrics): An instance of tf.keras.metrics.TruePositives which will work to calculate basic metrics.
        TN (tf.keras.metrics): An instance of tf.keras.metrics.TrueNegatives which will work to calculate basic metrics.
        FP (tf.keras.metrics): An instance of tf.keras.metrics.FalsePositives which will work to calculate basic metrics.
        FN (tf.keras.metrics): An instance of tf.keras.metrics.FalseNegatives which will work to calculate basic metrics.
        AUC (tf.keras.metrics): An instance of tf.keras.metrics.AUC which will work to calculate basic metrics.
        gen_loss (tf.keras.metrics): An instance of tf.keras.metrics.Mean which will work to calculate basic metrics.
        disc_loss (tf.keras.metrics): An instance of tf.keras.metrics.Mean which will work to calculate basic metrics.
        train_data (tf.data.Dataset): An instance of tf.data.Dataset containing the train data for the model.
        test_data (tf.data.Dataset): An instance of tf.data.Dataset containing the test data for the model.
        normal_class (Integer): An integer indicating which class will be the normal class, if control (0) or parkinson (1) patients.
    """
    random.seed(opts["seed"])
    np.random.seed(opts["seed"])
    tf.random.set_seed(opts["seed"])

    experiment_path, experiment_id = experiment_folder_path(opts["save_path"], opts["isize"], opts["nc"])
    metric_save_path = get_metrics_path(experiment_path)
    outputs_path = get_outputs_path(experiment_path)

    save_readme(experiment_path, readme_template, experiment_id, kfold)

    TP.reset_states()
    TN.reset_states()
    FP.reset_states()
    FN.reset_states()
    AUC.reset_states()
    gen_loss.reset_states()
    disc_loss.reset_states()

    gen_model, disc_model = get_model(opts["isize"], opts["nz"], opts["nc"], opts["ngf"], opts["extra_layers"])
    gen_opt = tf.keras.optimizers.Adam(learning_rate=opts["lr"], beta_1=opts["beta_1"], beta_2=opts["beta_2"])
    disc_opt = tf.keras.optimizers.Adam(learning_rate=opts["lr"], beta_1=opts["beta_1"], beta_2=opts["beta_2"])

    train_metrics_csv = open(os.path.join(metric_save_path,"train.csv"), "w+")
    train_metrics_csv.write("epoch,gen_error,disc_error,accuracy,precision,recall,specificity,f1_score,auc\n")

    test_metrics_csv = open(os.path.join(metric_save_path,"test.csv"), "w+")
    test_metrics_csv.write("epoch,accuracy,precision,recall,specificity,f1_score,auc\n")

    for epoch in range(opts["epochs"]):

        # Save the models every 1000 epochs
        if epoch % 1000 == 0 or epoch + 1 == opts["epochs"]:
            save_models(gen_model, disc_model, experiment_path, epoch)

        for step, xyi in enumerate(train_data):
            err_g, err_d, fake_images, latent_i, latent_o, feat_real, feat_fake = train_step(xyi[0], opts["w_gen"])

            if err_d < 1e-5 or tf.abs(err_d - disc_loss.result().numpy()) < 1e-5:
                reinit_model(disc_model)

            acc, pre, rec, spe, f1, auc = get_metrics(epoch, step, metric_save_path, xyi, normal_class, latent_i, latent_o, TP, TN, FP, FN, AUC, err_g, err_d)

            gen_loss.update_state(err_g)
            disc_loss.update_state(err_d)
            gen_error = gen_loss.result().numpy()
            disc_error = disc_loss.result().numpy()

            # Save the latent vectors, videos and errors in the last epoch and every 500 epochs
            if epoch % 1000 == 0 or epoch + 1 == opts["epochs"]:
                save_model_results(xyi, fake_images, latent_i, latent_o, feat_real, feat_fake, outputs_path, True)

        # Save train metrics
        train_metrics_csv.write("{e},{loss_g},{loss_d},{acc},{pre},{rec},{spe},{f1},{auc}\n".format(
            e = epoch,
            loss_g = gen_error,
            loss_d = disc_error,
            acc = acc,
            pre = pre,
            rec = rec,
            spe = spe,
            f1 = f1,
            auc = auc
        ))
        TP.reset_states()
        TN.reset_states()
        FP.reset_states()
        FN.reset_states()
        AUC.reset_states()
        gen_loss.reset_states()
        disc_loss.reset_states()
        
        del xyi
        del err_g
        del err_d
        del fake_images
        del latent_i
        del latent_o
        del feat_real
        del feat_fake

        for step, xyi in enumerate(test_data):
            fake_images, latent_i, latent_o, feat_real, feat_fake = test_step(xyi[0])

            acc, pre, rec, spe, f1, auc = get_metrics(epoch, step, metric_save_path, xyi, normal_class, latent_i, latent_o, TP, TN, FP, FN, AUC)

            # Save the latent vectors, videos and errors in the last epoch and every 500 epochs
            if epoch % 1000 == 0 or epoch + 1 == opts["epochs"]:
                save_model_results(xyi, fake_images, latent_i, latent_o, feat_real, feat_fake, outputs_path, False)

        # Save test metrics
        test_metrics_csv.write("{e},{acc},{pre},{rec},{spe},{f1},{auc}\n".format(
            e = epoch,
            acc = acc,
            pre = pre,
            rec = rec,
            spe = spe,
            f1 = f1,
            auc = auc
        ))
        TP.reset_states()
        TN.reset_states()
        FP.reset_states()
        FN.reset_states()
        AUC.reset_states()

    train_metrics_csv.close()
    test_metrics_csv.close()
    
    ######################### Save final models ###############################
    save_models(gen_model, disc_model, experiment_path)
    
    ######################### Deleting the used model ###############################
    del gen_model
    del disc_model
    del train_data
    del test_data
    del gen_opt
    del disc_opt
    del xyi
    del fake_images
    del latent_i
    del latent_o
    del feat_real
    del feat_fake
    tf.keras.backend.clear_session()
    gc.collect()