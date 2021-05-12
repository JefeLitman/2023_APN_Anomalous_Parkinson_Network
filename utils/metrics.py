"""This file contains different metrics to use with the models.
Version: 1.0
Made by: Edgar Rangel
"""

import tensorflow as tf
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

def shapiroWilks_test(data):
    """Function that execute the Shapiro-Wilks Test for normality where the null hypothesis is 'x comes from 
    a normal distribution' given the data and return the p-value obtained.
    Args:
        data (Array): An 1D array of data containing the values to be tested.
    """
    return stats.shapiro(data).pvalue

def dagostinoPearson_test(data):
    """Function that execute the D'Agostino-Pearson Test for normality where the null hypothesis is 'x comes from 
    a normal distribution' given the data and return the p-value obtained.
    Args:
        data (Array): An 1D array of data containing the values to be tested.
    """
    return stats.normaltest(data).pvalue

def levene_test(data_x, data_y):
    """Function that execute the Levene Test for homoscedasticity where the null hypothesis is 'x and y variances 
    are the same' given the x and y data and return the p-value obtained.
    Args:
        data_x (Array): An 1D array of data containing the values of x to be tested.
        data_y (Array): An 1D array of data containing the values of y to be tested.
    """
    return stats.levene(data_x, data_y, center='mean').pvalue

def bartlett_test(data_x, data_y):
    """Function that execute the Barlett Test for homoscedasticity where the null hypothesis is 'x and y variances 
    are the same' given the x and y data (both must follow a normal distribution) and return the p-value obtained.
    Args:
        data_x (Array): An 1D array of data containing the values of x to be tested.
        data_y (Array): An 1D array of data containing the values of y to be tested.
    """
    return stats.bartlett(data_x, data_y).pvalue

def fOneWay_test(data_x, data_y):
    """Function that execute the F (Fisher) Test where the null hypothesis if 'x and y means are the same' 
    given the x and y data and return the p-value obtained.
    Args:
        data_x (Array): An 1D array of data containing the values of x to be tested.
        data_y (Array): An 1D array of data containing the values of y to be tested.
    """
    return stats.f_oneway(data_x, data_y).pvalue
