"""
Defines loss functions to be used in package.
"""
from sklearn.metrics import hinge_loss
import numpy as np


def normalized_hinge_loss_1(predictor, x, y, labels):
    """
    Uses 1 - yz for hinge_loss, where z is the predicted label. Normalizes
    by dividing hinge_loss by 2.

    :param predictor:
    :param x:
    :param y:
    :param labels:
    :return:
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


def normalized_hinge_loss_2(predictor, x, y, labels):
    """
    Uses probability of y=1 label for calculating hinge_loss. Normalizes by dividing by 2.

    :param predictor:
    :param x:
    :param y:
    :param labels:
    :return:
    """

    # reshape x
    x_reshape = x.reshape(1, -1)

    # get probability of y=1
    y_true = np.full(shape=len(x_reshape), fill_value=y, dtype=np.int)
    y_prob = predictor.predict_proba(x_reshape)
    y_prob_second_label = np.array([y_prob.item(1)])

    # calculate loss
    loss = hinge_loss(y_true, y_prob_second_label, labels=labels)
    normalized = loss / 2.0  # to ensure loss is in [0,1]

    return normalized


def normalized_hinge_loss_3(predictor, x, y, labels):
    """
    Uses decision_function for calculating hinge_loss. Normalizes by dividing by 4.

    :param predictor:
    :param x:
    :param y:
    :param labels:
    :return:
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
