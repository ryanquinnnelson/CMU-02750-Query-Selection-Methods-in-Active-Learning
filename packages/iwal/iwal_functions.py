from scipy.stats import bernoulli


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


def _sum_losses(h, selected, loss, labels):
    """
    Sums losses over set of labeled elements.
    :param h:
    :param selected:
    :param loss:
    :param labels:
    :return:
    """
    total = 0
    for x, y_true, c in selected:
        # calculate loss for this sample
        y_predict = h.predict(x)
        curr_loss = loss(y_true, y_predict, labels=labels)
        iwal_loss = c * curr_loss

        # update total
        total += iwal_loss

    return total


# ?? difference between loss function L() vs l()
def _get_min_hypothesis(hypothesis_space, selected, loss, labels):
    """
    Calculates the minimum hypothesis in the hypothesis space, given a set of labeled samples with weights. Minimum is defined using the following equation:

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
def iwal_query(x_t, y_t, selected, rejection_threshold, history, hypothesis_space, loss, labels, **kwargs):
    """
    This function implements a single query IWAL algorithm from the paper by Beygelzimer et al. See
    https://arxiv.org/pdf/0812.4952.pdf

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
    :param kwargs: Dictionary of arbitrary arguments that may be required by rejection_threshold().
    :return: Instance (scikit-learn model) of hypothesis space that is optimal at current time step.
    """

    # derive probability of requesting label for x_t
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
