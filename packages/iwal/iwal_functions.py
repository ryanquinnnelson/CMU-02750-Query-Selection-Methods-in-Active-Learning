from scipy.stats import bernoulli
import numpy as np


def _append_history(history, x_t, y_t, p_t, q_t):
    """
    Adds new sample to given query history dictionary.

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

#?? predict_proba for log_loss vs pred_decision for hinge_loss
def _sum_losses(h, selected, loss, labels):
    """
    Sums losses over set of labeled elements.

    Uses decision_function() for hinge_loss.
    Would require predict_proba for log_loss. Need to refactor to allow both.
    :param h:
    :param selected:
    :param loss:
    :param labels:
    :return:
    """
    total = 0
    for x, y_true, c in selected:
        # calculate loss for this sample
        y_predict = h.decision_function(x)
        curr_loss = loss(y_true, y_predict, labels=labels)
        iwal_loss = c * curr_loss

        # update total
        total += iwal_loss

    return total


# ?? difference between loss function L() vs l()
def _get_min_hypothesis(hypothesis_space, selected, loss, labels):
    """
    Calculates the minimum hypothesis in the hypothesis space, given a set of labeled samples with weights. Minimum is
    defined using the following formula:

    h_t = argmin_{h in H} SUM_{x,y,c in S} c * l(h(x),y)

    :param hypothesis_space:
    :param selected:
    :param loss:
    :param labels:
    :return:
    """
    min_loss = 10000
    min_h = None

    # consider each model in hypothesis space
    for i in range(len(hypothesis_space)):

        # sum losses over all labeled elements
        h = hypothesis_space[i]
        curr_loss = _sum_losses(h, selected, loss, labels)

        # update minimum loss
        if curr_loss < min_loss:
            min_loss = curr_loss
            min_h = hypothesis_space[i]

    return min_h


# ?? how to choose Q_t? use Bernoulli distribution?
# ?? no y_t in parameters but listed in documentation
# ?? no S in parameters but is in paper
# ?? confirm h is list of models? Algorithm 1 requires argmin over all h in H...
def iwal_query(x_t, y_t, selected, rejection_threshold, history, hypothesis_space, loss, labels, **additional):
    """
    This function implements Algorithm 1 IWAL (subroutine rejection-threshold) from the paper by Beygelzimer et al. See
    https://arxiv.org/pdf/0812.4952.pdf.

    :param loss: Python function which calculates loss for a given hypothesis.
    :param labels: List of possible labels for the data set.
    :param x_t: Sample currently being considered for labelling.
    :param y_t: True label for sample.
    :param selected: List representing samples chosen for labeling. Each element in the list is a tuple (x,y,c), where
    x is the data, y is the label, and c is 1/p_t, where p_t is the rejection threshold probability for this sample.
    :param rejection_threshold: Python function which calculates the rejection threshold probability.
    :param history: Dictionary of query history. Must contain the keys 'X', 'y', 'p', 'Q'. The value of each key is a
    List containing the following:
            X -- sampled data points
            y -- labels matching to data points
            p -- rejection probabilities matching to data points
            Q -- coin flips matching to data points
    :param hypothesis_space: A list of scikit-learn models (i.e. sklearn.linear_model.LogisticRegression)
    :param additional: Dictionary of arbitrary arguments that may be required by rejection_threshold().
    :return: Instance (scikit-learn model) of hypothesis space that is optimal at current time step.
    """

    # derive probability of requesting label for x_t
    if additional:
        p_t = rejection_threshold(x_t, history, additional['kwargs'])  # pass additional arguments
    else:
        p_t = rejection_threshold(x_t, history)

    # flip a coin using derived probability
    Q_t = bernoulli.rvs(p_t)

    # save query in history
    _append_history(history, x_t, y_t, p_t, Q_t)

    # choose actions based on flip
    if Q_t == 1:  # label is requested
        c_t = 1.0 / p_t
        selected.append((x_t, y_t, c_t))  # add to set of selected samples

    # select model with least loss
    h_t = _get_min_hypothesis(hypothesis_space, selected, loss, labels)

    return h_t


def _loss_difference(y_true, y_pred_i, y_pred_j, loss, labels):
    """
    Calculates the difference in loss between two predictions.

    :param y_true:
    :param y_pred_i:
    :param y_pred_j:
    :param labels:
    :param loss:
    :return:
    """
    loss_i = loss(y_true, y_pred_i, labels=labels)
    loss_j = loss(y_true, y_pred_j, labels=labels)
    return loss_i - loss_j


def _bootstrap_probability(p_min, max_loss_difference):
    """
    Calculates the rejection threshold probability using the following formula:

    p_t = p_min + (1 - p_min) * max_loss_difference

    :param p_min:
    :param max_loss_difference:
    :return:
    """
    return p_min + (1 - p_min) * max_loss_difference


# ?? don't use loss parameter defined in the parameters
# ?? what to return if hypothesis space or labels are empty
# ?? how to know it is working correctly
def bootstrap(x, history, additional_args):
    """
    This function implements the bootstrap rejection threshold subroutine defined in 7.2. Bootstrap instantiation of
    IWAL from the paper by Beygelzimer et al. See https://arxiv.org/pdf/0812.4952.pdf. Calculates rejection threshold
    probability for a single sample using the following formula:

    p_t = p_min + (1 - p_min) * {max_{y, h_i, h_j} L(h_i(x), y) - L(h_j(x), y)}

    :param history:
    :param x:
    :param additional_args:
    :return:
    """
    # set additional arguments
    hypothesis_space = additional_args['H']
    loss = additional_args['loss']
    labels = additional_args['labels']
    p_min = additional_args['p_min']

    # find max loss difference
    max_loss_difference = -10000

    # consider every pair of models in the hypothesis space combined with every label
    for i in range(len(hypothesis_space)):
        for j in range(len(hypothesis_space)):
            for label in labels:

                # calculate loss difference between models
                y_true = np.full(shape=len(x), fill_value=label, dtype=np.int)
                y_pred_i = hypothesis_space[i].predict(x)
                y_pred_j = hypothesis_space[j].predict(x)

                curr_difference = _loss_difference(y_true, y_pred_i, y_pred_j, loss, labels)

                # update max
                if curr_difference > max_loss_difference:
                    max_loss_difference = curr_difference

    # calculate p_t
    p_t = _bootstrap_probability(p_min, max_loss_difference)
    return p_t
