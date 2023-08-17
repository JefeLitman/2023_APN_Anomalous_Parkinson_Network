"""This file contains methods to print into a log file with the calculation of metrics in training and eval for GANomaly model.
https://arxiv.org/abs/1805.06725
Version: 1.1
Made by: Edgar Rangel
"""

import os
from .losses import l2_loss_batch
from .processing import min_max_scaler
from utils.metrics import accuracy, precision, recall, specificity, f1_score

def get_metrics(epoch, step, experiment_path, xyi, normal_class, latent_i, latent_o, TP, TN, FP, FN, AUC, err_g=None, err_d=None):
    """This function calculate and save in the log file path the status and performance of the network while doing training or evaluation. This method return a tuple with the model metrics in the following order.
    Args:
        epoch (Integer): The epoch number.
        step (Integer): The step number in the epoch.
        experiment_path (String): A string with the folder path in where the log file will be written.
        xyi (Tuple[Tensor]): A tuple with (videos, labels, patient_ids, videos_ids) elements in that respective order.
        normal_class (Integer): An integer indicating which class will be the normal class, if control (0) or parkinson (1) patients.
        latent_i (Tensor): A tensor with the latent vector Zg of the model.
        latent_o (Tensor): A tensor with the latent vector Z'g of the model.
        TP (tf.keras.metrics): An instance of tf.keras.metrics.TruePositives which will work to calculate basic metrics.
        TN (tf.keras.metrics): An instance of tf.keras.metrics.TrueNegatives which will work to calculate basic metrics.
        FP (tf.keras.metrics): An instance of tf.keras.metrics.FalsePositives which will work to calculate basic metrics.
        FN (tf.keras.metrics): An instance of tf.keras.metrics.FalseNegatives which will work to calculate basic metrics.
        AUC (tf.keras.metrics): An instance of tf.keras.metrics.AUC which will work to calculate basic metrics.
        err_g (Decimal): Put this parameter when the model is in training phase for generator error.
        err_d (Decimal): Put this parameter when the model is in training phase for discriminator error.
    """
    anomaly_scores = min_max_scaler(l2_loss_batch(latent_i, latent_o), -1, 0, 1, -1)[0].numpy()

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

    text_to_print = "Epoch: {i} - Train Step: {j}".format(i = epoch + 1, j = step + 1)
    if err_g != None and err_d != None:
        text_to_print += "\nGenerator error: {loss_g}\nDiscriminator error: {loss_d}".format(loss_g = err_g, loss_d = err_d)
        log_name = "train.log"
    else:
        text_to_print = "Epoch: {i} - Test Step: {j}".format(i = epoch + 1, j = step + 1)
        log_name = "test.log"
    text_to_print += "\nAccuracy: {acc}\nPrecision: {pre}\nRecall: {rec}\nSpecificity: {spe}\nF1_Score: {f1}\nAUC: {auc}\n {line}".format(
        acc = acc,
        pre = pre,
        rec = rec,
        spe = spe,
        f1 = f1,
        auc = auc,
        line = '='*30
    )
    print(text_to_print)
    with open(os.path.join(experiment_path, log_name), 'a+') as f:
        print(text_to_print, file=f)
    return (acc, pre, rec, spe, f1, auc)
