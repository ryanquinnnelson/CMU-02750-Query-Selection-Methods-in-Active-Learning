from scipy.stats import bernoulli
import numpy as np
from typing import Any
from sklearn.metrics import hinge_loss, log_loss


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


def _loss_difference_hinge_loss(y_true: np.ndarray, pred_decision_a: np.ndarray, pred_decision_b: np.ndarray,
                                labels: list) -> float:
    """
    Calculates the difference in loss between a pair of prediction decisions, using hinge_loss.

    :param y_true:
    :param pred_decision_a:
    :param pred_decision_b:
    :param labels:
    :return:
    """
    loss_a = hinge_loss(y_true, pred_decision_a, labels=labels)
    loss_b = hinge_loss(y_true, pred_decision_b, labels=labels)

    return loss_a - loss_b


def _loss_difference(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, labels: list,
                     loss_function: str) -> float:
    """
    Calculates the difference in loss between a pair of prediction decisions, using given loss_function.

    :param y_true:
    :param y_pred_a:
    :param y_pred_b:
    :param labels:
    :param loss_function:
    :return:
    """
    if loss_function == 'hinge_loss':
        diff = _loss_difference_hinge_loss(y_true, y_pred_a, y_pred_b, labels)
    else:
        raise NotImplementedError('Function does not support loss_function: ' + loss_function)

    return diff

def _bootstrap_probability(p_min: float, max_loss_difference: float) -> float:
    """
    Calculates the rejection threshold probability using the following formula:

    p_t = p_min + (1 - p_min) * max_loss_difference

    :param p_min:
    :param max_loss_difference:
    :return:
    """
    return p_min + (1 - p_min) * max_loss_difference


def _get_prediction(x: np.ndarray, h: Any, loss_function: str) -> np.ndarray:
    """
    Calculates the appropriate prediction, given the loss_function.

    :param x:
    :param h:
    :param loss_function:
    :return:
    """
    if loss_function == 'hinge_loss':
        pred= h.decision_function(x)
    elif loss_function == 'log_loss':
        pred= h.predict_proba(x)
    else:
        raise NotImplementedError('Function does not support loss_function: ' + loss_function)

    return pred


# ?? what loss function to use? don't use loss parameter defined in the parameters
# ?? what to return if hypothesis space or labels are empty
# ?? how to know it is working correctly
def _bootstrap(x: np.ndarray, hypothesis_space: list, labels: list, p_min: float, loss_function: str) -> float:
    """
    This function implements the bootstrap rejection threshold subroutine defined in 7.2. Bootstrap instantiation of
    IWAL from the paper by Beygelzimer et al. See https://arxiv.org/pdf/0812.4952.pdf. Calculates rejection threshold
    probability for a single sample using the following formula:

    p_t = p_min + (1 - p_min) * {max_{y, h_i, h_j} L(h_i(x), y) - L(h_j(x), y)}

    :param x:
    :param hypothesis_space:
    :param labels:
    :param p_min:
    :param loss_function:
    :return:
    """
    max_diff = -10000

    # consider every pair of models in the hypothesis space combined with every label
    for i in range(len(hypothesis_space)):
        for j in range(len(hypothesis_space)):
            for label in labels:

                # based on loss_function, get appropriate prediction to calculate loss
                h_i = hypothesis_space[i]
                h_j = hypothesis_space[j]
                y_pred_i = _get_prediction(x, h_i, loss_function)
                y_pred_j = _get_prediction(x, h_j, loss_function)

                # calculate loss difference between models
                y_true = np.full(shape=len(x), fill_value=label, dtype=np.int)
                diff = _loss_difference(y_true, y_pred_i, y_pred_j, labels, loss_function)  # revisit later

                # update max
                if diff > max_diff:
                    max_diff = diff

    # calculate rejection threshold probability
    p_t = _bootstrap_probability(p_min, max_diff)
    return p_t


def _calculate_loss(y_true: np.ndarray, y_pred: np.ndarray, labels: list, loss_function: str) -> float:
    """
    Calculates loss for a given prediction using given loss_function.
    :param y_true:
    :param y_pred:
    :param labels:
    :param loss_function:
    :return:
    """
    if loss_function == 'hinge_loss':
        loss = hinge_loss(y_true, y_pred, labels=labels)
    elif loss_function == 'log_loss':
        loss = log_loss(y_true, y_pred, labels=labels)
    else:
        raise NotImplementedError('Function does not support loss_function: ' + loss_function)

    return loss


# ?? all at once?
#?? sum hinge loss per sample, or need to use all samples at once?
#?? predict_proba for log_loss vs pred_decision for hinge_loss
def _sum_losses(h: Any, selected: list, labels: list, loss_function: str) -> float:
    """
    Sums losses over set of labeled elements.

    :param h:
    :param selected:
    :param labels:
    :param loss_function:
    :return:
    """
    total = 0.0
    for x, y_true, c in selected:

        # calculate loss for this sample
        y_predict = _get_prediction(x, h, loss_function)
        curr_loss = _calculate_loss(y_true, y_predict, labels, loss_function)
        weighted_loss = c * curr_loss

        # update total
        total += weighted_loss

    return total


# ?? difference between loss function L() vs l()
def _get_min_hypothesis(hypothesis_space: list, selected: list, labels: list, loss_function: str) -> Any:

    min_loss = 10000
    min_h = None

    # consider each model in hypothesis space
    for i in range(len(hypothesis_space)):

        # sum losses over all labeled elements
        h = hypothesis_space[i]
        curr_loss = _sum_losses(h, selected, labels, loss_function)

        # update minimum loss
        if curr_loss < min_loss:
            min_loss = curr_loss
            min_h = h

    return min_h


# ?? how to choose Q_t? use Bernoulli distribution?
# ?? no y_t in parameters but listed in documentation
# ?? no S in parameters but is in paper
# ?? confirm h is list of models? Algorithm 1 requires argmin over all h in H...
def iwal_query_bootstrap(x_t:np.ndarray, y_t:np.ndarray,  hypothesis_space: list, history: dict, selected: list, labels: list, loss_function: str='hinge_loss',p_min: float=0.1 ) -> Any:
    """
    This function implements Algorithm 1 IWAL (subroutine rejection-threshold) from the paper by Beygelzimer et al and
    uses the bootstrap subroutine defined in 7.2. Bootstrap instantiation of IWAL. See
    https://arxiv.org/pdf/0812.4952.pdf.

    :param x_t:
    :param y_t:
    :param hypothesis_space:
    :param history:
    :param selected:
    :param labels:
    :param loss_function:
    :param p_min:
    :return:
    """
    # calculate probability of requesting label for x_t
    p_t = _bootstrap(x_t, hypothesis_space, labels, p_min, loss_function)

    # flip a coin using derived probability
    Q_t = bernoulli.rvs(p_t)

    # save query in history
    _append_history(history, x_t, y_t, p_t, Q_t)

    # choose actions based on flip
    if Q_t == 1:  # label is requested
        c_t = 1.0 / p_t
        selected.append((x_t, y_t, c_t))  # add to set of selected samples

    # select model with least loss
    h_t = _get_min_hypothesis(hypothesis_space, selected, labels, loss_function)

    return h_t



#
#
# """
#     Calculates the minimum hypothesis in the hypothesis space, given a set of labeled samples with weights. Minimum is
#     defined using the following formula:
#
#     h_t = argmin_{h in H} SUM_{x,y,c in S} c * l(h(x),y)
#     """

# def iwal_query(x_t, y_t, selected, rejection_threshold, history, hypothesis_space, loss, labels, **additional):
#     """
#     :param loss: Python function which calculates loss for a given hypothesis.
#     :param labels: List of possible labels for the data set.
#     :param x_t: Sample currently being considered for labelling.
#     :param y_t: True label for sample.
#     :param selected: List representing samples chosen for labeling. Each element in the list is a tuple (x,y,c), where
#     x is the data, y is the label, and c is 1/p_t, where p_t is the rejection threshold probability for this sample.
#     :param rejection_threshold: Python function which calculates the rejection threshold probability.
#     :param history: Dictionary of query history. Must contain the keys 'X', 'y', 'p', 'Q'. The value of each key is a
#     List containing the following:
#             X -- sampled data points
#             y -- labels matching to data points
#             p -- rejection probabilities matching to data points
#             Q -- coin flips matching to data points
#     :param hypothesis_space: A list of scikit-learn models (i.e. sklearn.linear_model.LogisticRegression)
#     :param additional: Dictionary of arbitrary arguments that may be required by rejection_threshold().
#     :return: Instance (scikit-learn model) of hypothesis space that is optimal at current time step.
#     """
