"""
Implements rejection-threshold functions for the Importance Weighting Active Learning (IWAL) algorithm defined in
the paper by Beygelzimer et al. See https://arxiv.org/pdf/0812.4952.pdf.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hinge_loss, log_loss


# done, tested
def _bootstrap_loss_function(predictor, x_t, y_true, labels):
    """

    :param predictor:
    :param x_t:
    :param y_true:
    :param labels:
    :return:
    """

    # get probability
    y_prob = predictor.predict_proba(x_t)
    loss = log_loss(y_true, y_prob, labels=labels)
    return loss


# done, tested
def _bootstrap_check_losses(loss_i,loss_j):
    """

    :param loss_i:
    :param loss_j:
    :return:
    """
    if loss_i < 0.0 or loss_i > 1.0 or loss_j < 0.0 or loss_j > 1.0:
        raise ValueError('Loss must be within range [0,1]:', loss_i, loss_j)


# done, tested
def _bootstrap_calc_max_loss(x_t, predictors, labels, loss_function):
    """
    Uses supplied loss_function to calculate the max loss difference.
    :param x_t:
    :param predictors:
    :param labels:
    :param loss_function:
    :return:
    """

    max_diff = -10000

    # consider every pair of models in the hypothesis space combined with every label
    for i in range(len(predictors)):
        for j in range(len(predictors)):
            for label in labels:

                h_i = predictors[i]
                h_j = predictors[j]

                # calculate loss difference
                loss_i = loss_function(h_i, x_t, label, labels)
                loss_j = loss_function(h_j,x_t,  label, labels)
                _bootstrap_check_losses(loss_i, loss_j)
                diff = loss_i - loss_j

                # update max difference
                if diff > max_diff:
                    max_diff = diff

    return max_diff


# done, tested
def _bootstrap_combine_p_min_and_max_loss(p_min, max_loss_difference):
    """
    Performs final calculation for rejection threshold probability p_t using the following formula:

    p_t = p_min + (1 - p_min) * max_loss_difference
    :param p_min:
    :param max_loss_difference:
    :return:
    """
    return p_min + (1 - p_min) * max_loss_difference


# done, need to test second case
def _bootstrap_calculate_p_t(x_t, predictors, labels, p_min, loss_function):
    """
    Calculates the rejection threshold probability p_t.
    :param x_t:
    :param predictors:
    :param labels:
    :param p_min:
    :param loss_function:
    :return:
    """

    # calculate max loss difference
    if loss_function:
        max_loss_diff = _bootstrap_calc_max_loss(x_t, predictors, labels, loss_function)
    else:
        max_loss_diff = _bootstrap_calc_max_loss(x_t, predictors, labels, _bootstrap_loss_function)

    # use max loss difference to calculate p_t
    p_t = _bootstrap_combine_p_min_and_max_loss(p_min, max_loss_diff)

    return p_t


# done, tested
def _bootstrap_y_has_all_labels(y,labels):
    """

    :param y:
    :param labels:
    :return:
    """

    for label in labels:
        if label not in y:
            return False

    return True


# done, tested
def _bootstrap_select_iid_training_set(X, y, labels):
    """
    Selects n random samples from the given data set, with replacement, where n is equal to the length of the data set.

    :param X:
    :param y:
    :return:
    """
    n = X.shape[0]

    # confirm y contains all labels expected in the data set
    y_contains_all_labels = _bootstrap_y_has_all_labels(y,labels)
    if not y_contains_all_labels:
        raise ValueError('y does not contain all labels expected in the data set.')

    # build a training set that contains
    training_has_all_labels = False
    while not training_has_all_labels:
        indexes = np.random.choice(n, n, replace=True)  # select n indices from a range of 0 to n-1
        training_has_all_labels = _bootstrap_y_has_all_labels(y[indexes],labels)

    return X[indexes], y[indexes]


# done, tested
def _bootstrap_reshape_history(X,y):
    """
    Combines list of separate samples in history to create a single data set.

    :param X:
    :param y:
    :return:
    """
    X_arr = np.concatenate(X, axis=0)
    y_arr = np.concatenate(y, axis=0)
    return X_arr,y_arr


# done, tested
def _bootstrap_select_history(history, bootstrap_size):
    """

    :param history:
    :param bootstrap_size:
    :return:
    """
    X_history = history['X'][:bootstrap_size]
    y_history = history['y'][:bootstrap_size]
    return X_history,y_history


# done, tested
def _bootstrap_train_predictors(history,bootstrap_size,num_predictors, labels):
    """
    Trains all predictors in the hypothesis space using bootstrapping.

    :param history:
    :param bootstrap_size:
    :param num_predictors:
    :param labels:
    :return:
    """

    # select training set to be used for bootstrapping
    X_history,y_history = _bootstrap_select_history(history, bootstrap_size)
    X, y = _bootstrap_reshape_history(X_history,y_history)

    # train predictors
    predictors = []
    for i in range(num_predictors):
        X_train, y_train = _bootstrap_select_iid_training_set(X, y, labels)
        lr = LogisticRegression().fit(X_train, y_train)
        predictors.append(lr)

    return predictors


# done, need to test second case
def bootstrap(x_t, history, bootstrap_size=10, num_predictors=10, labels=[0,1], p_min=0.1, loss_function=None):
    """
    This function implements Algorithm 3 from the paper by Beygelzimer et al. See https://arxiv.org/pdf/0812.4952.pdf.
    Uses bootstrapping to generate a hypothesis space and calculates rejection threshold probability p_t for unlabeled
    sample x_t.

    Note 1 - Bootstrapping process:
    To generate a diverse hypothesis space, each predictor is trained on a set of examples selected i.i.d. (at random
    with replacement) from the set of samples in history. At the time of bootstrapping, the number of samples in history
    is equal to the bootstrap size.
    :param x_t:
    :param history:
    :param bootstrap_size:
    :param num_predictors:
    :param labels:
    :param p_min:
    :param loss_function:
    :return:
    """
    # actions depend on current training size and bootstrap size
    training_size = len(history['X'])
    if training_size < bootstrap_size:  # bootstrapping process is not complete

        p_t = 1.0  # sample will be used for training and should be added to selected set

    else:
        # create bootstrapped committee of predictors
        predictors = _bootstrap_train_predictors(history,bootstrap_size,num_predictors,labels)

        # calculate probability
        p_t = _bootstrap_calculate_p_t(x_t, predictors, labels, p_min, loss_function)

    return p_t
