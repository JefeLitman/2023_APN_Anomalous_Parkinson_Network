"""This file contains different metrics to use with the models.
Version: 1.6
Made by: Edgar Rangel
"""

import numpy as np
import tensorflow as tf
from scipy import stats
from .common import repeat_vector_to_size

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
    if tp + tn + fp + fn == 0:
        return 0
    else:
        return (tp+tn)/(tp+tn+fp+fn)

def precision(tp, fp):
    """Function to calculate the precision obtained given the true positives and false positives.
    Args:
        tp (Integer): An integer specifying the quantity of true positives.
        fp (Integer): An integer specifying the quantity of false positives.
    """
    if tp + fp == 0:
        return 0
    else:
        return tp/(tp+fp)

def recall(tp, fn):
    """Function to calculate the recall obtained given the true positives and false negatives.
    Args:
        tp (Integer): An integer specifying the quantity of true positives.
        fn (Integer): An integer specifying the quantity of false negatives.
    """
    if tp + fn == 0:
        return 0
    else:
        return tp/(tp+fn)

def specificity(tn, fp):
    """Function to calculate the specificity obtained given the true negatives and false positives.
    Args:
        tn (Integer): An integer specifying the quantity of true negatives.
        fp (Integer): An integer specifying the quantity of false positives.
    """
    if tn + fp == 0:
        return 0
    else:
        return tn/(tn+fp)

def f1_score(tp, fp, fn):
    """Function to calculate the f1 score obtained given the true positives and false positives and negatives.
    Args:
        tp (Integer): An integer specifying the quantity of true positives.
        fp (Integer): An integer specifying the quantity of false positives.
        fn (Integer): An integer specifying the quantity of false negatives.
    """
    if tp + fp + fn == 0:
        return 0
    else:
        return 2*tp/(2*tp + fp + fn)

def get_AUC():
    """Function that return tf.keras.metrics Instance for AUC calculation."""
    return tf.keras.metrics.AUC(name="AUC")

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

def all_metrics_curve(y_true, y_pred, num_thresholds=200):
    """Function that calculate all metrics with thresholds contained between the min value of y_pred to the max value of y_pred.
    Args:
        y_true (Array): An 1D array of data containing the true values of classes.
        y_pred (Array): An 1D array of data containing the predicted values of classes.
        num_thresholds (Integer): How much integers will be evaluated between the range of min and max of y_pred.
    """
    accs = []
    pres = []
    recs = []
    spes = []
    f1s = []
    thresholds = np.linspace(np.min(y_pred), np.max(y_pred), num_thresholds)
    for t in thresholds:
        tp = np.count_nonzero(np.logical_and(y_true, (y_pred > t)))
        tn = np.count_nonzero(np.logical_and(np.logical_not(y_true), (y_pred <= t)))
        fp = np.count_nonzero(np.logical_and(np.logical_not(y_true), (y_pred > t)))
        fn = np.count_nonzero(np.logical_and(y_true, (y_pred <= t)))
        accs.append(accuracy(tp, tn, fp, fn))
        pres.append(precision(tp, fp))
        recs.append(recall(tp, fn))
        spes.append(specificity(tn, fp))
        f1s.append(f1_score(tp, fp, fn))

    return np.r_[accs], np.r_[pres], np.r_[recs], np.r_[spes], np.r_[f1s], thresholds

def __chiSquare_test__(data_experimental, data_theorical, alpha=0.05):
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

def __brownForsythe_test__(data_x, data_y, alpha=0.05):
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

def __levene_test__(data_x, data_y, alpha=0.05):
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

def homocedasticity_level(*classes):
    """Function that execute the Levene and Brown-Forsythe Test for homoscedasticity for every possible pair of arrays in each class array given. It returns a value between 0 and 1 indicating how much equally in variance the data are.
    Args:
        classes (List[Array]): Each class parameter you give must be a List of arrays, where each element in the list is a group of the class and each value in the array is a sample for that group in that class.
    """
    for c in classes:
        if not isinstance(c, list) and len(c) > 0:
            raise AssertionError("Every parameter in the method must be a native List element of python with at least one array inside.")
    all_samples = []
    for i, c in enumerate(classes):
        for group in c:
            all_samples.append((i, group))
    
    homo_level = []
    for i, group_1 in enumerate(all_samples[:-1]):
        for group_2 in all_samples[i+1:]:
            if group_1[0] == group_2[0]:
                prefix = 0
            else:
                prefix = 1
            data_1 = np.r_[sorted(group_1[1])]
            data_2 = np.r_[sorted(group_2[1])]
            homo_level += [
                abs(prefix - int(__brownForsythe_test__(data_1, data_2))), 
                abs(prefix - int(__levene_test__(data_1, data_2)))
            ]
    return np.mean(homo_level)

def shapeness_level(*classes, seed):
    """Function that execute the ChiSquare test in both directions for shapeness or equality in distribution for every possible pair of arrays in each class array given. It returns a value between 0 and 1 indicating how much equally in distribution the data are.
    Args:
        classes (List[Array]): Each class parameter you give must be a List of arrays, where each element in the list is a group of the class and each value in the array is a sample for that group in that class.
        seed (Integer): Integer to be used as seed to enable the replicability of the randomization permutation.
    """
    for c in classes:
        if not isinstance(c, list) and len(c) > 0:
            raise AssertionError("Every parameter in the method must be a native List element of python with at least one numpy array inside.")
    all_samples = []
    for i, c in enumerate(classes):
        for group in c:
            all_samples.append((i, group))
    
    class_level = []
    for i, group_1 in enumerate(all_samples[:-1]):
        for group_2 in all_samples[i+1:]:
            if group_1[0] == group_2[0]:
                prefix = 0
            else:
                prefix = 1
            if group_1[1].shape[0] > group_2[1].shape[0]:
                data_1 = np.r_[sorted(group_1[1])]
                data_2 = np.r_[sorted(repeat_vector_to_size(group_2[1], group_1[1].shape[0], seed))]
            elif group_1[1].shape[0] < group_2[1].shape[0]:
                data_1 = np.r_[sorted(repeat_vector_to_size(group_1[1], group_2[1].shape[0], seed))]
                data_2 = np.r_[sorted(group_2[1])]
            else:
                data_1 = np.r_[sorted(group_1[1])]
                data_2 = np.r_[sorted(group_2[1])]
            chi_test_1 = __chiSquare_test__(data_1, data_2)
            chi_test_2 = __chiSquare_test__(data_2, data_1)
            class_level += [
                abs(prefix - int(chi_test_1[0])),
                abs(prefix - int(chi_test_2[0]))
            ]
    return np.mean(class_level)
