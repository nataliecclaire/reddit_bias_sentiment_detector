from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import matplotlib.pyplot as plt
'''
import keras.backend as K
from itertools import product
import tensorflow as tf
'''


def confusion_matrix_report(y_true, y_pred):
    """
    Prints good looking confusion matrix
    Parameters
    ----------
    y_true : pd.DataFrame or pd.Series or np.array
        One column DataFrame/ array or Series with true labels
    y_pred : pd.DataFrame or pd.Series or np.array
        One column DataFrame/ array or Series with predicted labels

    Returns
    -------
    Nothing, only for printing
    """
    cm, labels = confusion_matrix(y_true, y_pred), unique_labels(y_true, y_pred)
    column_width = max([len(str(x)) for x in labels] + [5])  # 5 is value length
    report = " " * column_width + " " + "{:_^{}}".format("Prediction", column_width * len(labels))+ "\n"
    report += " " * column_width + " ".join(["{:>{}}".format(label, column_width) for label in labels]) + "\n"
    for i, label1 in enumerate(labels):
        report += "{:>{}}".format(label1, column_width) + " ".join(["{:{}d}".format(cm[i, j], column_width) for j in range(len(labels))]) + "\n"
    return report


def cost_scorer_2(y, y_pred):
    """
    Calculates costs based on predefined cost matrix
    Parameters
    ----------
    y : pd.DataFrame or pd.Series or np.array
        One column DataFrame/ array or Series with true labels
    y_pred : pd.DataFrame or pd.Series or np.array
        One column DataFrame/ array or Series with predicted labels


    Returns
    -------
    costs for predictions
    """

    cost_matrix = [[0, -25], [-5, 5]]
    # ToDo adjust in case we want to use a cost matrix (e.g. higher cost for predicting pred as real)

    cost = np.multiply(cost_matrix, confusion_matrix(y, y_pred))

    return np.sum(cost)
