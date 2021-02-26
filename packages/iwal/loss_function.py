"""
Defines loss functions to be used in this implementation of the Importance Weighting Active Learning (IWAL) algorithm
from the paper by Beygelzimer et al. See https://arxiv.org/pdf/0812.4952.pdf.
"""
import numpy as np
from typing import Any
from sklearn.metrics import hinge_loss


def normalized_hinge_loss_1(predictor: Any, x: np.ndarray, y: Any, labels: list) -> float:
    """
    Calculates normalized hinge loss using the output of the predictor's predict() function. The predicted label is
    used as the second argument in the hinge_loss() function. Normalizes in [0,1] by dividing by 2.

    :param predictor: sklearn model
    :param x: (2,) numpy array representing sample data point.
    :param y: element representing sample label.
    :param labels: List of all possible labels for the data set.
    :return: Float representing hinge_loss, normalized in [0,1].
    """

    # reshape x
    x_reshape = x.reshape(1, -1)

    # get prediction
    y_true = np.full(shape=len(x_reshape), fill_value=y, dtype=np.int)
    y_pred = predictor.predict(x_reshape)

    # calculate loss
    loss = hinge_loss(y_true, y_pred, labels=labels)
    normalized = loss / 2.0  # to ensure loss is in [0,1]

    return normalized


def normalized_hinge_loss_2(predictor: Any, x: np.ndarray, y: Any, labels: list) -> float:
    """
    Calculates normalized hinge loss using the probability of the last label in the given list of labels, where
    probability is calculated using given predictor's predict_proba() function. The selected probability is used as the
     second argument in the hinge_loss() function. Normalizes in [0,1] by dividing by 2.

    :param predictor: sklearn model
    :param x: (2,) numpy array representing sample data point.
    :param y: element representing sample label.
    :param labels: List of all possible labels for the data set.
    :return: Float representing hinge_loss, normalized in [0,1].
    """

    # reshape x
    x_reshape = x.reshape(1, -1)

    # get probabilities of all labels
    y_prob = predictor.predict_proba(x_reshape)

    # select the probability of the last label
    num_labels = y_prob.shape[1]
    last_idx = num_labels - 1  # index of the probability for the last label
    y_prob_last_label = np.array([y_prob.item(last_idx)])

    # calculate loss
    y_true = np.full(shape=len(x_reshape), fill_value=y, dtype=np.int)
    loss = hinge_loss(y_true, y_prob_last_label, labels=labels)
    normalized = loss / 2.0  # to ensure loss is in [0,1]

    return normalized


def normalized_hinge_loss_3(predictor: Any, x: np.ndarray, y: Any, labels: list) -> float:
    """
    Calculates normalized hinge loss using the output of given predictor's decision_function() function as the second
    argument in the hinge_loss() function. Normalizes in [0,1] by dividing by 10.

    :param predictor: sklearn model
    :param x: (2,) numpy array representing sample data point.
    :param y: element representing sample label.
    :param labels: List of all possible labels for the data set.
    :return: Float representing hinge_loss, normalized in [0,1].
    """

    # reshape x
    x_reshape = x.reshape(1, -1)

    # get decision
    y_true = np.full(shape=len(x_reshape), fill_value=y, dtype=np.int)
    y_dec = predictor.decision_function(x_reshape)

    # calculate loss
    loss = hinge_loss(y_true, y_dec, labels=labels)
    normalized = loss / 10.0  # to ensure loss is in [0,1]

    return normalized
