"""This file contains different metrics to use with the models.
Version: 1.2
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

def dagostinoPearson_test(data, alpha=0.05):
    """Function that execute the D'Agostino-Pearson Test for normality where the null hypothesis is 'x comes from 
    a normal distribution' given the data and return a boolean for the null hypothesis.
    Args:
        data (Array): An 1D array of data containing the values to be tested.
        alpha (Float): A decimal value meaning the significance level, default is 0.05 for 5%.
    """
    p = stats.normaltest(data).pvalue
    if p < alpha:
        return False
    else: 
        return True 

def andersonDarling_test(data, alpha=0.05):
    """Function that execute the Anderson-Darling Test for normality where the null hypothesis is 'x comes from 
    a normal distribution' given the data and return a boolean for the null hypothesis.
    Args:
        data (Array): An 1D array of data containing the values to be tested.
        alpha (Float): A decimal value meaning the significance level, default is 0.05 for 5%.
    """
    test = stats.anderson(data)
    if alpha*100 in test.significance_level:
        idx = np.argwhere(test.significance_level == alpha*100)[0,0]
        if test.statistic > test.critical_values[idx]:
            return False
        else:
            return True
    else:
        raise AttributeError("The alpha given is not available in the Anderson-Darling significance levels.")

def shapiroWilks_test(data, alpha=0.05):
    """Function that execute the Shapiro-Wilks Test for normality where the null hypothesis is 'x comes from 
    a normal distribution' given the data and return a boolean for the null hypothesis.
    Args:
        data (Array): An 1D array of data containing the values to be tested.
        alpha (Float): A decimal value meaning the significance level, default is 0.05 for 5%.
    """
    p = stats.shapiro(data).pvalue
    if p < alpha:
        return False
    else: 
        return True 

def chiSquare_test(data_experimental, data_teorical=None, alpha=0.05):
    """Function that execute the chi Square Test. When no teorical data is given the null 
    hypothesis is 'x comes from a chi square distribution' or when teorical data is given 
    the null hypothesis is 'experimental data follow the teorical data frequencies or 
    distribution' and finally returns a boolean for the null hypethesis with the statistical 
    value of the test.
    Args:
        data_experimental (Array): An 1D array of data containing the values to be tested.
        data_teorical (Array): An 1D array of data containing the expected values to be compared.
        alpha (Float): A decimal value meaning the significance level, default is 0.05 for 5%.
    """
    if hasattr(data_teorical, "shape"):
        test = stats.chisquare(data_experimental, data_teorical)
    else:
        test = stats.chisquare(data_experimental)
    if test.pvalue < alpha:
        return False, test.statistic
    else: 
        return True, test.statistic 

def fOneWay_test(data_x, data_y, alpha=0.05):
    """Function that execute the F (Fisher) Test where the null hypothesis is 'x and y means are the same' 
    or 'x and y comes from the same distribution' given the x and y data and return a boolean for the 
    null hypothesis.
    Args:
        data_x (Array): An 1D array of data containing the values of x to be tested.
        data_y (Array): An 1D array of data containing the values of y to be tested.
        alpha (Float): A decimal value meaning the significance level, default is 0.05 for 5%.
    """
    p =  stats.f_oneway(data_x, data_y).pvalue
    if p < alpha:
        return False
    else: 
        return True 

def brownForsythe_test(data_x, data_y, alpha=0.05):
    """Function that execute the Brown-Forsythe Test for homoscedasticity where the null hypothesis is 
    'x and y variances are the same' given the x and y data and return a boolean for the 
    null hypothesis.
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
    """Function that execute the Levene Test for homoscedasticity where the null hypothesis is 'x and y 
    variances are the same' given the x and y data and return a boolean for the 
    null hypothesis.
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

def bartlett_test(data_x, data_y, alpha=0.05):
    """Function that execute the Barlett Test for homoscedasticity where the null hypothesis is 'x and y variances 
    are the same' given the x and y data (both must follow a normal distribution) and return a boolean for the 
    null hypothesis.
    Args:
        data_x (Array): An 1D array of data containing the values of x to be tested.
        data_y (Array): An 1D array of data containing the values of y to be tested.
        alpha (Float): A decimal value meaning the significance level, default is 0.05 for 5%.
    """
    p = stats.bartlett(data_x, data_y).pvalue
    if p < alpha:
        return False
    else: 
        return True 

def mannWhitney_test(data_x, data_y, alpha=0.05):
    """Function that execute the Mann-Whitney Test where the null hypothesis is 'x and y comes from the same 
    distribution' given the x and y data and return a boolean for the null hypothesis.
    Args:
        data_x (Array): An 1D array of data containing the values of x to be tested.
        data_y (Array): An 1D array of data containing the values of y to be tested.
        alpha (Float): A decimal value meaning the significance level, default is 0.05 for 5%.
    """
    p = stats.mannwhitneyu(data_x, data_y, alternative='two-sided').pvalue
    if p < alpha:
        return False
    else: 
        return True 

def kruskalWallis_test(data_x, data_y, alpha=0.05):
    """Function that execute the Kruskal-Wallis Test where the null hypothesis is 'x and y comes from the same 
    distribution' given the x and y data and return a boolean for the null hypothesis.
    Args:
        data_x (Array): An 1D array of data containing the values of x to be tested.
        data_y (Array): An 1D array of data containing the values of y to be tested.
        alpha (Float): A decimal value meaning the significance level, default is 0.05 for 5%.
    """
    p = stats.kruskal(data_x, data_y).pvalue
    if p < alpha:
        return False
    else: 
        return True 

def kolmogorovSmirnov_test(data_x, data_y, alpha=0.05):
    """Function that execute the Kolmogorov-Smirnov Test where the null hypothesis is 'x and y comes from the same 
    distribution' given the x and y data and return a boolean for the null hypothesis.
    Args:
        data_x (Array): An 1D array of data containing the values of x to be tested.
        data_y (Array): An 1D array of data containing the values of y to be tested.
        alpha (Float): A decimal value meaning the significance level, default is 0.05 for 5%.
    """
    p = stats.kstest(data_x, data_y).pvalue
    if p < alpha:
        return False
    else: 
        return True 