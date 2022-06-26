"""This file contains the loop for train and eval mode for GANomaly 3D model.
Version: 0.1
Made by: Edgar Rangel
"""

import os
import random
import numpy as np
import tensorflow as tf
from ..model import get_model
from ..hiperparameters import get_options
from ..utils.savers import save_models
from ..utils.steps import train_step, test_step
from ..utils.exp_docs import experiment_folder_path, get_metrics_path, get_outputs_path, save_readme

def exec_loop(readme_template, kfold, TP, TN, FP, FN, AUC, gen_loss, disc_loss, train_data, test_data, normal_class):
    """This function execute the loop for training and evaluation in each epoch for GANomaly 3D model. It doesn't return anything but it will be showing the results obtained in each epoch.
    Args:
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
    opts = get_options()

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
        if epoch + 1 % 1000 == 0 or epoch + 1 == opts["epochs"]:
            save_models(gen_model, disc_model, experiment_path, outputs_path, epoch)

        for step, xyi in enumerate(train_data):
            err_g, err_d, fake_images, latent_i, latent_o, feat_real, feat_fake = train_step(xyi[0], opts["w_gen"])

            if err_d < 1e-5 or tf.abs(err_d - disc_loss.result().numpy()) < 1e-5:
                reinit_model(disc_model)

            anomaly_scores = tf.math.reduce_mean(tf.math.pow(tf.squeeze(latent_i-latent_o), 2), axis=1)
            anomaly_scores = (anomaly_scores - tf.reduce_min(anomaly_scores)) / (
                tf.reduce_max(anomaly_scores) - tf.reduce_min(anomaly_scores)
            )
            if normal_class == 1:
                anomaly_scores = 1 - anomaly_scores

            TP.update_state(xyi[1], anomaly_scores)
            TN.update_state(xyi[1], anomaly_scores)
            FP.update_state(xyi[1], anomaly_scores)
            FN.update_state(xyi[1], anomaly_scores)
            AUC.update_state(xyi[1], anomaly_scores)
            gen_loss.update_state(err_g)
            disc_loss.update_state(err_d)
            acc = accuracy(TP.result().numpy(), TN.result().numpy(), FP.result().numpy(), FN.result().numpy())
            pre = precision(TP.result().numpy(), FP.result().numpy())
            rec = recall(TP.result().numpy(), FN.result().numpy())
            spe = specificity(TN.result().numpy(), FP.result().numpy())
            f1 = f1_score(TP.result().numpy(), FP.result().numpy(), FN.result().numpy())
            auc = AUC.result().numpy()
            gen_error = gen_loss.result().numpy()
            disc_error = disc_loss.result().numpy()

            clear_output(wait=True)
            print_metrics(epoch, step, acc, pre, rec, spe, f1, auc, err_g, err_d)

            # Save the latent vectors, videos and errors in the last epoch and every 500 epochs
            if epoch + 1 == epochs or epoch % 1000 == 0:
                save_latent_vectors(tf.squeeze(latent_i).numpy(), xyi[1].numpy(), xyi[2].numpy(), outputs_path[0], True)
                save_latent_vectors(tf.squeeze(latent_o).numpy(), xyi[1].numpy(), xyi[2].numpy(),  outputs_path[1], True)
                save_latent_vectors(tf.reshape(feat_real, [xyi[0].shape[0], -1]).numpy(), xyi[1].numpy(), xyi[2].numpy(), outputs_path[2], True)
                save_latent_vectors(tf.reshape(feat_fake, [xyi[0].shape[0], -1]).numpy(), xyi[1].numpy(), xyi[2].numpy(),  outputs_path[3], True)

                batch_frames = np.r_[[min_max_scaler(i, 0, 0, 255, 0)[0].numpy() for i in xyi[0]]]
                save_frames(
                    batch_frames, 
                    xyi[1].numpy(), 
                    xyi[2].numpy(), 
                    xyi[4].numpy(), 
                    xyi[3].numpy(), 
                    outputs_path[4],
                    True
                )
                batch_frames = np.r_[[min_max_scaler(i, 0, 0, 255, 0)[0].numpy() for i in fake_images]]
                save_frames(
                    batch_frames, 
                    xyi[1].numpy(), 
                    xyi[2].numpy(), 
                    xyi[4].numpy(), 
                    xyi[3].numpy(), 
                    outputs_path[5],
                    True
                )
                batch_frames = np.r_[[min_max_scaler(i, 0, 0, 255, 0)[0].numpy() for i in tf.abs(xyi[0] - fake_images)]]
                save_frames(
                    batch_frames, 
                    xyi[1].numpy(), 
                    xyi[2].numpy(), 
                    xyi[4].numpy(), 
                    xyi[3].numpy(), 
                    outputs_path[6],
                    True
                )

                save_errors(l2_loss_batch(feat_real, feat_fake), xyi[1].numpy(), outputs_path[7], True)
                save_errors(l1_loss_batch(xyi[0], fake_images), xyi[1].numpy(), outputs_path[8], True)
                save_errors(l2_loss_batch(latent_i, latent_o), xyi[1].numpy(), outputs_path[9], True)

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

            anomaly_scores = tf.math.reduce_mean(tf.math.pow(tf.squeeze(latent_i-latent_o), 2), axis=1)
            anomaly_scores = (anomaly_scores - tf.reduce_min(anomaly_scores)) / (
                tf.reduce_max(anomaly_scores) - tf.reduce_min(anomaly_scores)
            )
            if normal_class == 1:
                anomaly_scores = 1 - anomaly_scores

            TP.update_state(xyi[1], anomaly_scores)
            TN.update_state(xyi[1], anomaly_scores)
            FP.update_state(xyi[1], anomaly_scores)
            FN.update_state(xyi[1], anomaly_scores)
            AUC.update_state(xyi[1], anomaly_scores)
            acc = accuracy(TP.result().numpy(), TN.result().numpy(), FP.result().numpy(), FN.result().numpy())
            pre = precision(TP.result().numpy(), FP.result().numpy())
            rec = recall(TP.result().numpy(), FN.result().numpy())
            spe = specificity(TN.result().numpy(), FP.result().numpy())
            f1 = f1_score(TP.result().numpy(), FP.result().numpy(), FN.result().numpy())
            auc = AUC.result().numpy()

            clear_output(wait=True)
            print_metrics(epoch, step, acc, pre, rec, spe, f1, auc)

            # Save the latent vectors, videos and errors in the last epoch and every 500 epochs
            if epoch + 1 == epochs or epoch % 1000 == 0:
                save_latent_vectors(tf.squeeze(latent_i).numpy(), xyi[1].numpy(), xyi[2].numpy(), outputs_path[0], False)
                save_latent_vectors(tf.squeeze(latent_o).numpy(), xyi[1].numpy(), xyi[2].numpy(), outputs_path[1], False)
                save_latent_vectors(tf.reshape(feat_real, [xyi[0].shape[0], -1]).numpy(), xyi[1].numpy(), xyi[2].numpy(), outputs_path[2], False)
                save_latent_vectors(tf.reshape(feat_fake, [xyi[0].shape[0], -1]).numpy(), xyi[1].numpy(), xyi[2].numpy(), outputs_path[3], False)

                batch_frames = np.r_[[min_max_scaler(i, 0, 0, 255, 0)[0].numpy() for i in xyi[0]]]
                save_frames(
                    batch_frames, 
                    xyi[1].numpy(), 
                    xyi[2].numpy(), 
                    xyi[4].numpy(), 
                    xyi[3].numpy(), 
                    outputs_path[4],
                    False
                )
                batch_frames = np.r_[[min_max_scaler(i, 0, 0, 255, 0)[0].numpy() for i in fake_images]]
                save_frames(
                    batch_frames, 
                    xyi[1].numpy(), 
                    xyi[2].numpy(), 
                    xyi[4].numpy(), 
                    xyi[3].numpy(), 
                    outputs_path[5],
                    False
                )
                batch_frames = np.r_[[min_max_scaler(i, 0, 0, 255, 0)[0].numpy() for i in tf.abs(xyi[0] - fake_images)]]
                save_frames(
                    batch_frames, 
                    xyi[1].numpy(), 
                    xyi[2].numpy(), 
                    xyi[4].numpy(), 
                    xyi[3].numpy(), 
                    outputs_path[6],
                    False
                )

                save_errors(l2_loss_batch(feat_real, feat_fake), xyi[1].numpy(), outputs_path[7], False)
                save_errors(l1_loss_batch(xyi[0], fake_images), xyi[1].numpy(), outputs_path[8], False)
                save_errors(l2_loss_batch(latent_i, latent_o), xyi[1].numpy(), outputs_path[9], False)

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
    for i in sorted(os.listdir(experiment_path)):
        if "gen_model" in i:
            os.remove(os.path.join(experiment_path, i))
        elif "disc_model" in i:
            os.remove(os.path.join(experiment_path, i))
    gen_model.save(os.path.join(experiment_path,"gen_model.h5"), include_optimizer=False, save_format='h5')
    disc_model.save(os.path.join(experiment_path,"disc_model.h5"), include_optimizer=False, save_format='h5')
    
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