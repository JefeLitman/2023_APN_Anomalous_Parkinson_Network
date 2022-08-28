"""This file contains different metrics to use with the models.
Version: 1.4
Made by: Edgar Rangel
"""

import tensorflow as tf
import numpy as np
from scipy import stats

def get_true_positives():
    """Function that return tf.keras.metrics Instance for true positives calculation."""
    return tf.keras.metrics.TruePositives(name="true_positives")

def get_true_negatives():
    """Function that return tf.keras.metrics Instance for true negatives calculation."""
    return tf.keras.metrics.TrueNegatives(name="true_negatives")

def get_false_positives():
    """Function that return tf.keras.metrics Instance for false positives calculation."""
    return tf.keras.metrics.FalsePositives(name="false_positives")

def get_false_negatives():
    """Function that return tf.keras.metrics Instance for false negatives calculation."""
    return tf.keras.metrics.FalseNegatives(name="false_negatives")

def get_mean():
    """Function that return tf.keras.metrics Instance for mean calculation."""
    return tf.keras.metrics.Mean(name="mean")

def accuracy(tp, tn, fp, fn):
    """Function to calculate the accuracy obtained given the true and false positives and negatives.
    Args:
        tp (Integer): An integer specifying the quantity of true positives.
        tn (Integer): An integer specifying the quantity of true negatives.
        fp (Integer): An integer specifying the quantity of false positives.
        fn (Integer): An integer specifying the quantity of false negatives.
    """
    return (tp+tn)/(tp+tn+fp+fn)

def precision(tp, fp):
    """Function to calculate the precision obtained given the true positives and false positives.
    Args:
        tp (Integer): An integer specifying the quantity of true positives.
        fp (Integer): An integer specifying the quantity of false positives.
    """
    return tp/(tp+fp)

def recall(tp, fn):
    """Function to calculate the recall obtained given the true positives and false negatives.
    Args:
        tp (Integer): An integer specifying the quantity of true positives.
        fn (Integer): An integer specifying the quantity of false negatives.
    """
    return tp/(tp+fn)

def specificity(tn, fp):
    """Function to calculate the specificity obtained given the true negatives and false positives.
    Args:
        tn (Integer): An integer specifying the quantity of true negatives.
        fp (Integer): An integer specifying the quantity of false positives.
    """
    return tn/(tn+fp)

def f1_score(tp, fp, fn):
    """Function to calculate the f1 score obtained given the true positives and false positives and negatives.
    Args:
        tp (Integer): An integer specifying the quantity of true positives.
        fp (Integer): An integer specifying the quantity of false positives.
        fn (Integer): An integer specifying the quantity of false negatives.
    """
    return 2*tp/(2*tp + fp + fn)

def get_AUC():
    """Function that return tf.keras.metrics Instance for AUC calculation."""
    return tf.keras.metrics.AUC(name="AUC")

def precision_recall_curve(y_true, y_pred, num_thresholds=200):
    """Function that calculate different precisions and recalls with thresholds contained between the min value of y_pred to the max value of y_pred.
    Args:
        y_true (Array): An 1D array of data containing the true values of classes.
        y_pred (Array): An 1D array of data containing the predicted values of classes.
        num_thresholds (Integer): How much integers will be evaluated between the range of min and max of y_pred.
    """
    precisions = []
    recalls = []
    thresholds = np.linspace(np.min(y_pred), np.max(y_pred), num_thresholds)
    for t in thresholds:
        tp = np.count_nonzero(np.logical_and(y_true, (y_pred > t)))
        fp = np.count_nonzero(np.logical_and(np.logical_not(y_true), (y_pred > t)))
        fn = np.count_nonzero(np.logical_and(y_true, (y_pred <= t)))
        if tp+fp == 0:
            precisions.append(0)
        else:
            precisions.append(precision(tp, fp))
        if tp + fn == 0:
            recalls.append(0)
        else:
            recalls.append(recall(tp, fn))
    return np.r_[precisions], np.r_[recalls], thresholds

def tpr_fpr_curve(y_true, y_pred, num_thresholds=200):
    """Function that calculate different TPR (True Positive Rate) and FPR (False Positive Rate) with thresholds contained between the min value of y_pred to the max value of y_pred.
    Args:
        y_true (Array): An 1D array of data containing the true values of classes.
        y_pred (Array): An 1D array of data containing the predicted values of classes.
        num_thresholds (Integer): How much integers will be evaluated between the range of min and max of y_pred.
    """
    tpr = []
    fpr = []
    thresholds = np.linspace(np.min(y_pred), np.max(y_pred), num_thresholds)
    for t in thresholds:
        tp = np.count_nonzero(np.logical_and(y_true, (y_pred > t)))
        tn = np.count_nonzero(np.logical_and(np.logical_not(y_true), (y_pred <= t)))
        fp = np.count_nonzero(np.logical_and(np.logical_not(y_true), (y_pred > t)))
        fn = np.count_nonzero(np.logical_and(y_true, (y_pred <= t)))
        if tp + fn == 0:
            tpr.append(0)
        else:
            tpr.append(tp / (tp + fn))
        if tn + fp == 0:
            fpr.append(0)
        else:
            fpr.append(fp / (fp + tn))
    return np.r_[tpr], np.r_[fpr], thresholds

def chiSquare_test(data_experimental, data_theorical, alpha=0.05):
    """Function that execute the chi Square Test. In this case the theorical data is required to test the null hypothesis of 'experimental data follow the theorical data frequencies or distribution' and finally returns a boolean for the null hypothesis with the statistical value of the test. This methods is based in scipy chisquare method but its applied by hand.
    Args:
        data_experimental (Array): An 1D array of data containing the values to be tested.
        data_teorical (Array): An 1D array of data containing the expected values to be compared.
        alpha (Float): A decimal value meaning the significance level, default is 0.05 for 5%.
    """
    terms = (data_experimental - data_theorical)**2 / data_theorical
    statistic = np.sum(terms)
    p_value = stats.chi2.sf(statistic, data_theorical.shape[0] - 1)
    if p_value < alpha:
        return False, statistic
    else: 
        return True, statistic 

def brownForsythe_test(data_x, data_y, alpha=0.05):
    """Function that execute the Brown-Forsythe Test for homoscedasticity where the null hypothesis is 'x and y variances are the same' given the x and y data and return a boolean for the null hypothesis.
    Args:
        data_x (Array): An 1D array of data containing the values of x to be tested.
        data_y (Array): An 1D array of data containing the values of y to be tested.
        alpha (Float): A decimal value meaning the significance level, default is 0.05 for 5%.
    """
    p = stats.levene(data_x, data_y, center='median').pvalue
    if p < alpha:
        return False
    else: 
        return True 

def levene_test(data_x, data_y, alpha=0.05):
    """Function that execute the Levene Test for homoscedasticity where the null hypothesis is 'x and y variances are the same' given the x and y data and return a boolean for the null hypothesis.
    Args:
        data_x (Array): An 1D array of data containing the values of x to be tested.
        data_y (Array): An 1D array of data containing the values of y to be tested.
        alpha (Float): A decimal value meaning the significance level, default is 0.05 for 5%.
    """
    p = stats.levene(data_x, data_y, center='mean').pvalue
    if p < alpha:
        return False
    else: 
        return True 

# def brownForsythe_test(*samples, alpha=0.05):
#     """Function that execute the Brown-Forsythe Test for homoscedasticity where the null hypothesis is 'samples variances are the same' and return a boolean for the null hypothesis. This methods is based in scipy levene method but its applied by hand.
#     Args:
#         samples (Arrays): A set of 1D arrays containing the data values of x to be tested.
#         alpha (Float): A decimal value meaning the significance level, default is 0.05 for 5%.
#     """
#     k = len(samples)
#     Ni = np.empty(k)
#     Yci = np.empty(k, 'd')

#     for j in range(k):
#         Ni[j] = len(samples[j])
#         Yci[j] = np.median(samples[j])
#     Ntot = np.sum(Ni, axis=0)

#     # compute Zij's
#     Zij = [None] * k
#     for i in range(k):
#         Zij[i] = abs(np.asarray(samples[i]) - Yci[i])

#     # compute Zbari
#     Zbari = np.empty(k, 'd')
#     Zbar = 0.0
#     for i in range(k):
#         Zbari[i] = np.mean(Zij[i], axis=0)
#         Zbar += Zbari[i] * Ni[i]

#     Zbar /= Ntot
#     numer = (Ntot - k) * np.sum(Ni * (Zbari - Zbar)**2, axis=0)

#     # compute denom_variance
#     dvar = 0.0
#     for i in range(k):
#         dvar += np.sum((Zij[i] - Zbari[i])**2, axis=0)

#     denom = (k - 1.0) * dvar

#     W = numer / denom
#     pval = stats.f.sf(W, k-1, Ntot-k, scale=np.mean(Ni)/4)  # 1 - cdf
#     if pval < alpha:
#         return False
#     else: 
#         return True 

# def levene_test(*samples, alpha=0.05):
#     """Function that execute the Levene Test for homoscedasticity where the null hypothesis is 'samples variances are the same' and return a boolean for the null hypothesis. This methods is based in scipy levene method but its applied by hand.
#     Args:
#         samples (Arrays): A set of 1D arrays containing the data values of x to be tested.
#         alpha (Float): A decimal value meaning the significance level, default is 0.05 for 5%.
#     """
#     k = len(samples)
#     Ni = np.empty(k)
#     Yci = np.empty(k, 'd')

#     for j in range(k):
#         Ni[j] = len(samples[j])
#         Yci[j] = np.mean(samples[j])
#     Ntot = np.sum(Ni, axis=0)

#     # compute Zij's
#     Zij = [None] * k
#     for i in range(k):
#         Zij[i] = abs(np.asarray(samples[i]) - Yci[i])

#     # compute Zbari
#     Zbari = np.empty(k, 'd')
#     Zbar = 0.0
#     for i in range(k):
#         Zbari[i] = np.mean(Zij[i], axis=0)
#         Zbar += Zbari[i] * Ni[i]

#     Zbar /= Ntot
#     numer = (Ntot - k) * np.sum(Ni * (Zbari - Zbar)**2, axis=0)

#     # compute denom_variance
#     dvar = 0.0
#     for i in range(k):
#         dvar += np.sum((Zij[i] - Zbari[i])**2, axis=0)

#     denom = (k - 1.0) * dvar

#     W = numer / denom
#     pval = stats.f.sf(W, k-1, Ntot-k, scale=np.mean(Ni)/4)  # 1 - cdf
#     if pval < alpha:
#         return False
#     else: 
#         return True 
