"""
Implements rejection-threshold functions for Importance Weighting Active Learning (IWAL) algorithm from the paper by
Beygelzimer et al. See https://arxiv.org/pdf/0812.4952.pdf.
"""
from packages.iwal.helper import calculate_hinge_loss


# done, tested
def _bootstrap_calculate_p_t(p_min, max_loss_difference):
    """
    Calculates the rejection threshold probability p_t using the following formula:

    p_t = p_min + (1 - p_min) * max_loss_difference

    :param p_min:
    :param max_loss_difference:
    :return:
    """
    return p_min + (1 - p_min) * max_loss_difference


# done, tested
def _bootstrap_calculate_max_loss_difference(x, hypothesis_space, labels, loss_difference_function):
    max_diff = -10000

    # consider every pair of models in the hypothesis space combined with every label
    for i in range(len(hypothesis_space)):
        for j in range(len(hypothesis_space)):
            for label in labels:

                h_i = hypothesis_space[i]
                h_j = hypothesis_space[j]

                diff = loss_difference_function(x, h_i, h_j, label, labels)

                if diff > max_diff:
                    max_diff = diff

    return max_diff


# done, tested
# ?? l() vs L()
def _bootstrap_ldf_hinge(x, h_i, h_j, label, labels):
    """
    Defines a function to calculates loss difference using hinge_loss.
    Calculates loss using the following formula: L(h_i(x), y) - L(h_j(x), y).

    :param x:
    :param h_i:
    :param h_j:
    :param y_true:
    :return:
    """

    loss_i = calculate_hinge_loss(x, h_i, label, labels)
    loss_j = calculate_hinge_loss(x, h_j, label, labels)

    return loss_i - loss_j


# done, tested
def _bootstrap_train_predictors(hypothesis_space, history):

    X = history['X']  # entire history so far consists of selected samples
    y = history['y']

    for h in hypothesis_space:
        h.fit(X, y)


# done, testable
# ?? what loss function to use? difference between loss function L() vs l()
def bootstrap(x_t, h_space, bootstrap_size, history, labels, p_min=0.1):
    """
    This function implements Algorithm 3 from the paper by Beygelzimer et al. See https://arxiv.org/pdf/0812.4952.pdf.
    Calculates rejection threshold probability for unlabeled sample x_t. Uses hinge_loss for loss calculations.

    :param x_t:
    :param h_space:
    :param bootstrap_size:
    :param history:
    :param labels:
    :param p_min:
    :return:
    """

    # determine current round of active learning
    t = len(history['X']) + 1  # zero length history indicates round 1

    if t <= bootstrap_size:
        p_t = 1.0

        # consider whether to also train predictors on the initial sample
        if t == bootstrap_size:
            _bootstrap_train_predictors(h_space, history)
    else:
        max_loss = _bootstrap_calculate_max_loss_difference(x_t, h_space, labels, _bootstrap_ldf_hinge)
        p_t = _bootstrap_calculate_p_t(p_min, max_loss)

    return p_t
