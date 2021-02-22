"""
Implements rejection-threshold functions for the Importance Weighting Active Learning (IWAL) algorithm defined in
the paper by Beygelzimer et al. See https://arxiv.org/pdf/0812.4952.pdf.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


# done, tested
def _bootstrap_check_losses(loss_i, loss_j):
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

    max_diff = -10000  # arbitrary starting value

    # consider every pair of models in the hypothesis space combined with every label
    for i in range(len(predictors)):
        for j in range(len(predictors)):
            for label in labels:

                h_i = predictors[i]
                h_j = predictors[j]

                # calculate loss difference
                loss_i = loss_function(h_i, x_t, label, labels)
                loss_j = loss_function(h_j, x_t, label, labels)
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


# done
def _bootstrap_calculate_p_t(x_t, predictors, labels, p_min, loss_function):
    """
    Calculates the rejection threshold probability p_t.
    If predictors is empty, p_t is set to 1.
    :param x_t:
    :param predictors:
    :param labels:
    :param p_min:
    :param loss_function:
    :return:
    """

    if len(predictors) > 0:

        # calculate probability
        max_loss_diff = _bootstrap_calc_max_loss(x_t, predictors, labels, loss_function)
        p_t = _bootstrap_combine_p_min_and_max_loss(p_min, max_loss_diff)

    else:

        # default value is used
        p_t = 1.0

    return p_t


# done, tested
def _bootstrap_y_has_all_labels(y, labels):
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

    n = len(X)  # size of desired training set

    # build a training set that contains all labels expected in the data set
    training_has_all_labels = False
    while not training_has_all_labels:
        indexes = np.random.choice(n, n, replace=True)  # select n indices from a range of 0 to n-1
        training_has_all_labels = _bootstrap_y_has_all_labels(y[indexes], labels)

    return X[indexes], y[indexes]


# done, tested
def _bootstrap_select_bootstrap_training_set(history, bootstrap_size):
    """

    :param history:
    :param bootstrap_size:
    :return:
    """
    X_history = history['X'][:bootstrap_size]
    y_history = history['y'][:bootstrap_size]
    return X_history, y_history


# done, tested
def _bootstrap_train_predictors(history, bootstrap_size, num_predictors, labels):
    """
    Trains all predictors in the hypothesis space using bootstrapping. If training
    select selected from history does not contain all labels expected in the data, no predictors are trained.

    Note 1 - Bootstrapping process:
    To generate a diverse hypothesis space, each predictor is trained on a set of examples selected i.i.d. (at random
    with replacement) from the set of samples in history.

    :param history:
    :param bootstrap_size:
    :param num_predictors:
    :param labels:
    :return:
    """

    # limit training set to predetermined portion of history
    X_train, y_train = _bootstrap_select_bootstrap_training_set(history, bootstrap_size)

    predictors = []
    if _bootstrap_y_has_all_labels(y_train, labels):

        # train predictors
        for i in range(num_predictors):
            X_iid, y_iid = _bootstrap_select_iid_training_set(X_train, y_train, labels)
            lr = LogisticRegression().fit(X_iid, y_iid)
            predictors.append(lr)

    return predictors


# done
def bootstrap(x_t, history, loss_function, bootstrap_size=10, num_predictors=10, labels=[0, 1], p_min=0.1):
    """
    This function implements Algorithm 3 from the paper by Beygelzimer et al. See https://arxiv.org/pdf/0812.4952.pdf.
    Uses bootstrapping to generate a hypothesis space and calculates rejection threshold probability p_t for unlabeled
    sample x_t.
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
    if 'X' in history:
        training_size = len(history['X'])
    else:
        training_size = 0

    if training_size < bootstrap_size:  # bootstrapping process is not complete

        p_t = 1.0  # sample will be used for bootstrapping and should be added to selected set

    else:
        # create bootstrapped committee of predictors
        predictors = _bootstrap_train_predictors(history, bootstrap_size, num_predictors, labels)

        # calculate probability
        p_t = _bootstrap_calculate_p_t(x_t, predictors, labels, p_min, loss_function)

    return p_t
