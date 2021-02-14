from scipy.stats import bernoulli


def _append_history(history, x_t, y_t, p_t, q_t):
    """
    Adds new sample to given query history.

    :param history:
    :param x_t:
    :param y_t:
    :param p_t:
    :param q_t:
    :return:
    """
    # validate that history contains required keys
    if 'X' in history and 'y' in history and 'c' in history and 'Q' in history:
        history['X'].append(x_t)
        history['y'].append(y_t)
        history['c'].append(p_t)
        history['Q'].append(q_t)
    else:
        raise ValueError('history dictionary does not contain the required keys: X,y,c,Q')


# ?? seems wrong
# ?? difference between loss function L() vs l()
def _get_h_min(h, s, loss, labels):
    """
    Summary here.

    :param h: Hypothesis space. This is a List of scikit-learn objects.
    :param s: Set representing samples chosen for labeling, where each
    element in the set is a tuple {x,y,c}. c is 1/p, where p is the
    rejection threshold probability for this sample.
    :return: Object of scikit-learn model class H, that is optimal at
    current time step according to IWAL algorithm.
    """

    min_value = 10000
    min_h = None

    # consider each model in hypothesis space
    for i in len(h):

        # sum losses over all labeled elements
        curr_value = 0
        for x, y_true, c in s:
            # calculate loss
            y_pred = h[i].predict(x)
            curr_loss = loss(y_true, y_pred, labels=labels)
            iwal_loss = c * curr_loss
            curr_value += iwal_loss

        # update minimum
        if curr_value < min_value:
            min_value = curr_value
            min_h = h[i]

    return min_h


# ?? S vs history
# ?? how to choose Q_t? use Bernoulli distribution?
# ?? no y_t in parameters
# ?? no S in parameters
# ?? where is H defined? hypothesis space
# ?? H is not list of models?
def iwal_query(x_t, y_t, s, rejection_threshold, history, h, **kwargs):
    """
    This function implements a single query IWAL algorithm from the
    paper by Beygelzimer et al https://arxiv.org/pdf/0812.4952.pdf

    :param x_t: Sample currently being considered for labelling.
    :param y_t: True label for sample.
    :param s: Set representing samples chosen for labeling, where each
    element in the set is a tuple {x,y,c}. c is 1/p, where p is the rejection
    threshold probability for this sample.
    :param rejection_threshold: python function, accepts current data
    point x_t and some arbitrary arguments and returns probability of
    requesting a label for x_t.
    :param history: Dictionary of query history. history.keys() will return
    dict_keys(['X', 'y', 'p', 'Q']), where the value of each key is a List
    containing the following:
            X -- sampled data points
            y -- labels matching to data points
            p -- rejection probabilities matching to data points
            Q -- coin flips matching to data points
    :param h: scikit-learn model class, such as
    sklearn.linear_model.LogisticRegression, etc.
    :param kwargs: dictionary of arbitrary arguments that may be required
    by rejection_threshold().
    :return: Object of scikit-learn model class H that is optimal at current
     time step according to IWAL algorithm.
    """

    # derive probability of requesting label for x_t
    p_t = rejection_threshold(x_t, history)

    # flip a coin using derived probability
    Q_t = bernoulli.rvs(p_t)

    # record history
    _append_history(history, x_t, y_t, p_t, Q_t)

    # choose actions based on flip
    if Q_t == 1:  # label is requested
        c_t = 1.0 / p_t
        s.add((x_t, y_t, c_t))  # add to set of selected samples

    # select model with least loss
    # h_t = _get_h_min(h, s, 1)

    # return h_t
