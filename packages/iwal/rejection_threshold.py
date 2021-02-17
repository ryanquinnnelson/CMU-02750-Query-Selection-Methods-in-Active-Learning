"""
Implements rejection-threshold functions for the Importance Weighting Active Learning (IWAL) algorithm defined in
the paper by Beygelzimer et al. See https://arxiv.org/pdf/0812.4952.pdf.
"""
from packages.iwal.helper import calculate_hinge_loss
import numpy as np


# done, testable
def _bootstrap_calc_z_value(x, min_value, max_value):
    if max_value - min_value == 0.0:
        z_value = 0.0
    else:
        z_value = (x - min_value) / (max_value - min_value)

    return z_value


# done, testable
def _bootstrap_calc_max_loss_difference_internal(x, h_space, labels):
    """
    Uses a workaround to derived a max loss difference normalized to [0,1].

    Uses hinge_loss as the underlying loss function.

    :param x:
    :param h_space:
    :param labels:
    :return:
    """
    max_diff = -10000
    max_loss = -10000
    min_loss = 10000

    max_diff_i = None
    max_diff_j = None

    # consider every pair of models in the hypothesis space combined with every label
    for i in range(len(h_space)):
        for j in range(len(h_space)):
            for label in labels:

                h_i = h_space[i]
                h_j = h_space[j]

                # calculate loss difference
                loss_i = calculate_hinge_loss(x, h_i, label, labels)
                loss_j = calculate_hinge_loss(x, h_j, label, labels)
                diff = loss_i - loss_j

                # update max difference
                if diff > max_diff:
                    max_diff = diff
                    max_diff_i = loss_i
                    max_diff_j = loss_j

                # update min and max for normalization process
                if loss_i > max_loss:
                    max_loss = loss_i

                if loss_i < min_loss:
                    min_loss = loss_i

                if loss_j > max_loss:
                    max_loss = loss_j

                if loss_j < min_loss:
                    min_loss = loss_j

    # normalize losses which resulted in max loss difference
    normalized_i = _bootstrap_calc_z_value(min_loss, max_loss, max_diff_i)
    normalized_j = _bootstrap_calc_z_value(min_loss, max_loss, max_diff_j)
    return normalized_i - normalized_j


# done, testable
def _bootstrap_calc_max_loss_using_loss_function(x, h_space, labels, loss_function):
    """
    Uses supplied loss_function to calculate the max loss difference.

    :param x:
    :param h_space:
    :param labels:
    :param loss_function:
    :return:
    """

    max_diff = -10000

    # consider every pair of models in the hypothesis space combined with every label
    for i in range(len(h_space)):
        for j in range(len(h_space)):
            for label in labels:

                h_i = h_space[i]
                h_j = h_space[j]

                # calculate loss difference
                loss_i = loss_function(x, h_i, label, labels)
                loss_j = loss_function(x, h_j, label, labels)
                diff = loss_i - loss_j

                # update max difference
                if diff > max_diff:
                    max_diff = diff

    return max_diff


# done, testable
def _bootstrap_combine_p_min_and_max_loss(p_min, max_loss_difference):
    """
    Performs final calculation for rejection threshold probability p_t using the following formula:

    p_t = p_min + (1 - p_min) * max_loss_difference

    :param p_min:
    :param max_loss_difference: Maximum loss difference, normalized to [0,1].
    :return:
    """
    if 0.0 <= max_loss_difference <= 1.0:
        p_t = p_min + (1 - p_min) * max_loss_difference
    else:
        raise ValueError("Maximum loss difference must be normalized to [0,1]")

    return p_t


# done, testable
def _bootstrap_calculate_p_t(x_t, h_space, labels, loss_function, p_min):
    """
    Calculates the rejection threshold probability p_t.

    :param x_t:
    :param h_space:
    :param labels:
    :param loss_function:
    :param p_min:
    :return:
    """

    # calculate max loss difference
    if loss_function:
        standardized_max_loss_diff = _bootstrap_calc_max_loss_using_loss_function(x_t, h_space, labels, loss_function)

    else:
        standardized_max_loss_diff = _bootstrap_calc_max_loss_difference_internal(x_t, h_space, labels)

    # use max loss difference to calculate p_t
    p_t = _bootstrap_combine_p_min_and_max_loss(p_min, standardized_max_loss_diff)

    return p_t


# done, testable
def _bootstrap_select_iid_training_set(X, y):
    """
    Selects n random samples from the given data set, with replacement. n is equal to the length of the data set.

    :param X:
    :param y:
    :return:
    """
    n = X.shape[0]
    indexes = np.random.choice(n, n, replace=True)  # select n indices from a range of 0 to n-1
    return X[indexes], y[indexes]


# done, testable
def _bootstrap_reshape_history(history):
    """
    Combines list of separate samples in history to create a single data set.

    :param history:
    :return:
    """

    X = np.concatenate(history['X'], axis=0)
    y = np.concatenate(history['y'], axis=0)
    return X, y


# done, testable
def _bootstrap_train_predictors(h_space, history):
    """
    Trains all predictors in the hypothesis space using bootstrapping.

    Note the entire history so far consists of selected samples, and the entire history should be used as the
    bootstrapping training set.

    :param h_space:
    :param history:
    :return:
    """

    # create training set
    X, y = _bootstrap_reshape_history(history)

    # train all predictors
    for h in h_space:
        # train predictor using i.i.d. training set
        X_train, y_train = _bootstrap_select_iid_training_set(X, y)
        h.fit(X_train, y_train)


# done, testable
def bootstrap(x_t, h_space, bootstrap_size, history, labels, loss_function, p_min=0.1):
    """
    This function implements Algorithm 3 from the paper by Beygelzimer et al. See https://arxiv.org/pdf/0812.4952.pdf.
    Uses bootstrapping to generate a hypothesis space and calculates rejection threshold probability p_t for unlabeled
    sample x_t.

    Note 1 - Bootstrapping process:
    To generate a diverse hypothesis space, each predictor is trained on a set of examples selected i.i.d. (at random
    with replacement) from the set of samples in history. At the time of bootstrapping, the number of samples in history
    is equal to the bootstrap size.

    :param x_t:
    :param h_space:
    :param bootstrap_size:
    :param history:
    :param labels:
    :param loss_function: Defines a loss function L(h(x),y) which standardizes output in the range [0,1].
    :param p_min:
    :return:
    """
    # actions depend on current training size and bootstrap size
    training_size = len(history['X'])
    if training_size <= bootstrap_size:  # bootstrapping process is not complete

        p_t = 1.0  # sample will be used for training and should be added to selected set

        if training_size == bootstrap_size:  # training set meets desired size for bootstrapping
            _bootstrap_train_predictors(h_space, history)

    else:
        # trained predictors will be used to calculate p_t
        p_t = _bootstrap_calculate_p_t(x_t, h_space, labels, loss_function, p_min)

    return p_t
