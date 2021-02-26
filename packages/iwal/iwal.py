"""
Implements Importance Weighting Active Learning (IWAL) algorithm from the paper by Beygelzimer et al.
See https://arxiv.org/pdf/0812.4952.pdf.
"""
import numpy as np
import packages.iwal.rejection_threshold as rt
import packages.iwal.loss_function as lf
from typing import Any
from scipy.stats import bernoulli
from sklearn.linear_model import LogisticRegression


def _append_history(history: dict, x_t: np.ndarray, y_t: int, p_t: float, q_t: int) -> None:
    """
    Adds given sample to query history dictionary. If dictionary is empty, initializes key-value pairs for the four
    tracked categories.


    :param history: Dictionary containing query history. Expected to be empty or contain four keys: 'X','y','c','Q'.
                    'X' contains a numpy array (n,2) where each row represents a sample and n is the number of
                    queries made so far. 'y' contains a numpy array (n,) where each row represents a sample label.
                    'c' contains a list where each element represents the probability a sample was selected for
                    labeling. 'Q' contains a list where each element represents the 0-1 result of a coin flip.

    :param x_t: (2,) numpy array representing sample data point.
    :param y_t: (1,) numpy array representing sample label.
    :param p_t: Float representing probability of selecting sample for labeling.
    :param q_t: Integer result of coin flip deciding whether to select sample for labeling. 1 if selected, 0 otherwise.
    :return: None
    """

    if 'X' not in history:
        history['X'] = np.array([x_t])
    else:
        appended = np.append(history['X'], [x_t], axis=0)
        history['X'] = appended

    if 'y' not in history:
        history['y'] = np.array(y_t)
    else:
        appended = np.append(history['y'], y_t, axis=0)
        history['y'] = appended

    if 'c' not in history:
        history['c'] = [p_t]
    else:
        history['c'].append(p_t)

    if 'Q' not in history:
        history['Q'] = [q_t]
    else:
        history['Q'].append(q_t)


def _add_to_selected(selected, x_t, y_t, c_t) -> None:
    """
    Appends given sample to list of selected samples.

    :param selected: Dictionary containing all samples which were selected for labeling. Expected to be empty or
                     contain three keys: 'X','y','c'. 'X' contains a numpy array (n,2) where each row represents a
                     sample and n is the number of queries made so far. 'y' contains a numpy array (n,) where each row
                     represents a sample label. 'c' contains a list where each element represents the sample weight.
    :param x_t: (2,) numpy array representing sample data point
    :param y_t: (1,) numpy array representing sample label
    :param c_t: Float representing weight given to selected sample
    :return: None
    """
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


def _choose_flip_action(flip: int, selected: dict, x_t: np.ndarray, y_t: np.ndarray, p_t: float) -> None:
    """
    Takes an action depending on the outcome of a coin flip, where 1 indicates label is requested. If flip is 1,
    sets sample weight c_t and adds given sample to list of selected samples.

    Note about c_t: The paper defines c_t in two ways: (1) c_t=1/p_t; (2) c_t=p_min/p_t. This implementation uses 1/p_t.
    :param flip: Integer result of coin flip deciding whether to select sample for labeling. 1 if selected, 0 otherwise.
    :param selected: Dictionary containing all samples which were selected for labeling. Expected to be empty or
                     contain three keys: 'X','y','c'. 'X' contains a numpy array (n,2) where each row represents a
                     sample and n is the number of queries made so far. 'y' contains a numpy array (n,) where each row
                     represents a sample label. 'c' contains a list where each element represents the sample weight.
    :param x_t: (2,) numpy array representing sample data point
    :param y_t: (1,) numpy array representing sample label
    :param p_t: Float representing probability of selecting sample for labeling
    :return: none
    """

    if flip == 1:  # label is requested

        # define weight for this sample
        c_t = 1 / p_t

        # add to set of selected samples
        _add_to_selected(selected, x_t, y_t, c_t)


def _all_labels_in_selected(selected: dict, labels: list) -> bool:
    """
    Determines whether list of selected samples contains at least one sample for every given label.
    :param selected: Dictionary containing all samples which were selected for labeling. Expected to be empty or
                     contain three keys: 'X','y','c'. 'X' contains a numpy array (n,2) where each row represents a
                     sample and n is the number of queries made so far. 'y' contains a numpy array (n,) where each row
                     represents a sample label. 'c' contains a list where each element represents the sample weight.
    :param labels: List of all possible labels for the data set.
    :return: True if all labels are found, False otherwise.
    """
    for each in labels:
        if each not in selected['y']:
            return False
    return True


def iwal_query(x_t: np.ndarray, y_t: np.ndarray, history: dict, selected: dict, rejection_threshold: str,
               loss_function: Any = lf.normalized_hinge_loss_1, labels: list = [0, 1], p_min: float = 0.1) -> Any:
    """
    Performs Importance Weighting Active Learning (IWAL) for a single sample.

    :param x_t: (2,) numpy array representing sample data point.
    :param y_t: (1,) numpy array representing sample label.
    :param history: Dictionary containing query history. Expected to be empty or contain four keys: 'X','y','c','Q'.
                    'X' contains a numpy array (n,2) where each row represents a sample and n is the number of
                    queries made so far. 'y' contains a numpy array (n,) where each row represents a sample label.
                    'c' contains a list where each element represents the probability a sample was selected for
                    labeling. 'Q' contains a list where each element represents the 0-1 result of a coin flip.
    :param selected: Dictionary containing all samples which were selected for labeling. Expected to be empty or
                     contain three keys: 'X','y','c'. 'X' contains a numpy array (n,2) where each row represents a
                     sample and n is the number of queries made so far. 'y' contains a numpy array (n,) where each row
                     represents a sample label. 'c' contains a list where each element represents the sample weight.
    :param rejection_threshold: String defining the rejection_threshold subroutine to use with IWAL. The following
                                options are accepted: (1) bootstrap. Default value is 'bootstrap'.
    :param loss_function: Python function which calculates loss normalized to [0,1] range. The following signature is
                          expected: function_name(predictor:Any, x_t:np.ndarray, label:Any, labels:list) -> Float.
                          predictor: sklearn model;
                          x_t: (2,) numpy array representing sample data point;
                          label: single label from list of labels;
                          labels: List of all possible labels for the data set.
                          Default value is iwal.loss_function.normalized_hinge_loss_1.
    :param labels: List of all possible labels for the data set. Default value is [0,1].
    :param p_min: Minimum probability for selecting a sample for labeling. Default value is 0.1.
    :return: Predictor trained on the set of labeled samples.
    """
    # calculate probability of requesting label for x_t using chosen rejection threshold function
    if rejection_threshold == 'bootstrap':
        p_t = rt.bootstrap(x_t, history, loss_function=loss_function, p_min=p_min)
    else:
        raise NotImplementedError('Function does not support rejection_threshold:', rejection_threshold)

    # flip a coin using derived probability
    Q_t = bernoulli.rvs(p_t)

    # save query in history
    _append_history(history, x_t, y_t, p_t, Q_t)

    # choose actions based on flip
    _choose_flip_action(Q_t, selected, x_t, y_t, p_t)

    # select model with least loss
    if _all_labels_in_selected(selected, labels):
        h_t = LogisticRegression().fit(selected['X'], selected['y'], sample_weight=selected['c'])
    else:
        h_t = None  # optimal hypothesis can't be calculated

    return h_t
