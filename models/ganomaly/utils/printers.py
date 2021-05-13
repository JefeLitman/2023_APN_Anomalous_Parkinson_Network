"""This file contains methods to print the network metrics in training 
and inference for GANomaly models.
Version: 1.0
Made by: Edgar Rangel
"""

def print_metrics(epoch, step, acc, pre, rec, spe, f1, auc, err_g=None, err_d=None):
    """Function to print the values given in the parameters to show the status of 
    training or inference process for the model.
    Args:
        epoch (Integer): The epoch number.
        step (Integer): The step number in the epoch.
        acc (Decimal): The accuracy obtained in the step.
        pre (Decimal): The precision obtained in the step.
        rec (Decimal): The recall obtained in the step.
        spe (Decimal): The specificity obtained in the step.
        f1 (Decimal): The f1 score obtained in the step.
        auc (Decimal): The auc obtained in the step.
        err_g (Decimal): Put this parameter when the model is in training phase for generator error.
        err_d (Decimal): Put this parameter when the model is in training phase for discriminator error.
    """
    text_to_print = "Epoch: {i} - Train Step: {j}".format(i = epoch + 1, j = step + 1)
    if err_g != None and err_d != None:
        text_to_print += "\nGenerator error: {loss_g}\nDiscriminator error: {loss_d}".format(loss_g = err_g, loss_d = err_d)
    text_to_print += "Accuracy: {acc}\nPrecision: {pre}\nRecall: {rec}\nSpecificity: {spe}\nF1_Score: {f1}\nAUC: {auc}".format(
        acc = acc,
        pre = pre,
        rec = rec,
        spe = spe,
        f1 = f1,
        auc = auc
    )
    print(text_to_print)