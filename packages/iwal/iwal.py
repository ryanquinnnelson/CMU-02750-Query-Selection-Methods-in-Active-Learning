"""
Implements Importance Weighting Active Learning (IWAL) algorithm from the paper by Beygelzimer et al.
See https://arxiv.org/pdf/0812.4952.pdf.
"""
from typing import Any
import numpy as np
import packages.iwal.rejection_threshold as rt
from scipy.stats import bernoulli
from sklearn.linear_model import LogisticRegression


# done, tested2
def _append_history(history: dict, x_t: np.ndarray, y_t: int, p_t: float, q_t: int) -> None:
    """
    Adds given sample to given query history dictionary.

    :param history:
    :param x_t:
    :param y_t:
    :param p_t:
    :param q_t:
    :return:
    """
    if 'X' not in history:
        history['X'] = np.array([x_t])
    else:
        appended = np.append(history['X'], [x_t], axis=0)
        history['X'] = appended

    if 'y' not in history:
        history['y'] = np.array([y_t])
    else:
        appended = np.append(history['y'], [y_t], axis=0)
        history['y'] = appended

    if 'c' not in history:
        history['c'] = [p_t]
    else:
        history['c'].append(p_t)

    if 'Q' not in history:
        history['Q'] = [q_t]
    else:
        history['Q'].append(q_t)


# done, tested2
def _add_to_selected(selected, x_t, y_t, c_t):
    if 'X' not in selected:
        selected['X'] = np.array([x_t])
    else:
        appended = np.append(selected['X'], [x_t], axis=0)
        selected['X'] = appended

    if 'y' not in selected:
        selected['y'] = np.array(y_t)
    else:
        appended = np.append(selected['y'], y_t, axis=0)
        selected['y'] = appended

    if 'c' not in selected:
        selected['c'] = [c_t]
    else:
        selected['c'].append(c_t)


# done, tested2
def _choose_flip_action(flip: int, selected: dict, x_t: np.ndarray, y_t: np.ndarray, p_t: float, p_min: float) -> None:
    """
    Takes an action depending on the outcome of a coin flip, where 1 indicates label is requested.

    Note: The paper describes c_t in two ways. Once as 1/p_t. The other as p_min/p_t. I'm not sure whether using one
    version or the other will impact the results.
    :param flip:
    :param selected:
    :param x_t:
    :param y_t:
    :param p_t:
    :return:
    """

    if flip == 1:  # label is requested

        # define weight for this sample
        c_t = p_min / p_t

        # add to set of selected samples
        _add_to_selected(selected, x_t, y_t, c_t)


# done, tested2
def _all_labels_in_selected(selected, labels):
    """

    :param selected:
    :param labels:
    :return:
    """
    for each in labels:
        if each not in selected['y']:
            return False
    return True


# done, testable
def iwal_query(x_t: np.ndarray, y_t: np.ndarray, history: dict, selected: dict, rejection_threshold: str,
               labels: list = [0, 1], p_min: float = 0.1) -> Any:
    """

    :param x_t:
    :param y_t:
    :param history:
    :param selected:
    :param rejection_threshold:
    :param labels:
    :param p_min:
    :return:
    """
    # calculate probability of requesting label for x_t using chosen rejection threshold function
    if rejection_threshold == 'bootstrap':
        p_t = rt.bootstrap(x_t, history, p_min=p_min)
    else:
        raise NotImplementedError('Function does not support rejection_threshold:', rejection_threshold)

    # flip a coin using derived probability
    Q_t = bernoulli.rvs(p_t)

    # save query in history
    _append_history(history, x_t, y_t, p_t, Q_t)

    # choose actions based on flip
    _choose_flip_action(Q_t, selected, x_t, y_t, p_t, p_min)

    # select model with least loss
    if _all_labels_in_selected(selected, labels):
        h_t = LogisticRegression().fit(selected['X'], selected['y'], sample_weight=selected['c'])
    else:
        h_t = None  # optimal hypothesis can't be calculated

    return h_t
