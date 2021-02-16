"""
Implements Importance Weighting Active Learning (IWAL) algorithm from the paper by Beygelzimer et al.
See https://arxiv.org/pdf/0812.4952.pdf.
"""
from typing import Any
import numpy as np
import packages.iwal.rejection_threshold as rt
from scipy.stats import bernoulli
from packages.iwal.helper import calculate_hinge_loss


# done, testable
def _append_history(history: dict, x_t: np.ndarray, y_t: np.ndarray, p_t: float, q_t: int) -> None:
    """
    Adds given sample to given query history dictionary.

    :param history:
    :param x_t:
    :param y_t:
    :param p_t:
    :param q_t:
    :return:
    """
    # validate that history contains required keys
    if 'X' in history and 'y' in history and 'c' in history and 'Q' in history:

        # add to history
        history['X'].append(x_t)
        history['y'].append(y_t)
        history['c'].append(p_t)
        history['Q'].append(q_t)

    else:
        raise ValueError('history dictionary does not contain the required keys: X,y,c,Q')


# done, testable
def _choose_flip_action(flip: int, selected: list, x_t: np.ndarray, y_t: np.ndarray, p_t: float, p_min: float) -> None:
    """
    Takes an action depending on the outcome of a coin flip, where 1 indicates label is requested.
    :param flip:
    :param selected:
    :param x_t:
    :param y_t:
    :param p_t:
    :return:
    """

    if flip == 1:  # label is requested
        c_t = p_min / p_t
        selected.append((x_t, y_t, c_t))  # add to set of selected samples


# done, testable
# ?? difference between loss function L() vs l()
def _loss_summation_function(h: Any, selected: list, labels: list) -> float:
    total = 0.0
    for x, y, c in selected:
        loss = calculate_hinge_loss(x, h, y, labels)  # replace with uncoupled function
        weighted_loss = c * loss

        # update total
        total += weighted_loss
    return total


# done, testable
def _get_min_hypothesis(h_space: list, selected: list, labels: list, loss_summation_function: Any) -> Any:
    """
    Finds the min hypothesis h_t in the hypothesis space, given a set of labeled samples with weights. Minimum is
    defined using the following formula:

    h_t = argmin_{h in H} SUM_{(x,y,c) in S} c * l(h(x),y)

    where H is the hypothesis space, S is the set of labeled samples, and l() is the loss_summation_function.

    :param h_space:
    :param selected:
    :param labels:
    :param loss_summation_function:
    :return:
    """

    min_loss = 10000
    min_h = None

    # consider each model in hypothesis space
    for i in range(len(h_space)):

        # sum losses over all labeled elements
        h = h_space[i]
        loss = loss_summation_function(h, selected, labels)

        # update minimum loss
        if loss < min_loss:
            min_loss = loss
            min_h = h

    return min_h


# working
# ?? hinge_loss prefers {-1,1} for labels. Should we convert our labels to those?
def iwal_query(x_t: np.ndarray,
               y_t: np.ndarray,
               h_space: list,
               history: dict,
               selected: list,
               labels: list,
               rejection_threshold: str,
               b: int,
               p_min: float = 0.1) -> Any:
    # calculate probability of requesting label for x_t
    if rejection_threshold == 'bootstrap':
        p_t = rt.bootstrap(x_t, h_space, b, history, labels, p_min)
    else:
        raise NotImplementedError('Function does not support rejection_threshold:', rejection_threshold)

    # flip a coin using derived probability
    Q_t = bernoulli.rvs(p_t)

    # save query in history
    _append_history(history, x_t, y_t, p_t, Q_t)

    # choose actions based on flip
    _choose_flip_action(Q_t, selected, x_t, y_t, p_t, p_min)

    # select model with least loss
    h_t = _get_min_hypothesis(h_space, selected, labels, _loss_summation_function)

    return h_t