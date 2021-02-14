import packages.iwal.iwal_functions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hinge_loss, log_loss
import numpy as np


def test_append_history_dictionary_matches():
    history = {
        'X': [],
        'y': [],
        'c': [],
        'Q': []
    }

    x_t = [1, 2]
    y_t = 1
    p_t = 0.5
    q_t = 1

    packages.iwal.iwal_functions._append_history(history, x_t, y_t, p_t, q_t)

    assert x_t in history['X']
    assert y_t in history['y']
    assert p_t in history['c']
    assert q_t in history['Q']


def test__get_min_hypothesis_empty_hypothesis_space():
    hypothesis_space = []
    s = {(1, 1, 0.5)}
    labels = [0, 1]

    min_h = packages.iwal.iwal_functions._get_min_hypothesis(hypothesis_space, s, log_loss, labels)
    if min_h is None:
        assert True
    else:
        assert False


def test__get_min_hypothesis_empty_set():
    # example data set
    X = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
    y = [1, 0]

    lr = LogisticRegression().fit(X, y)
    hypothesis_space = [lr]
    s = set()
    labels = [0, 1]

    min_h = packages.iwal.iwal_functions._get_min_hypothesis(hypothesis_space, s, log_loss, labels)
    if min_h is lr:
        assert True
    else:
        assert False


def test__get_min_hypothesis_one_hypothesis():
    # example data set
    X = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
    y = [1, 0]
    lr = LogisticRegression().fit(X, y)
    hypothesis_space = [lr]

    # example labeled set
    x_t = [[3, 1]]
    y_t = [1]
    c_t = 0.1
    s = [(x_t, y_t, c_t)]

    labels_t = [0, 1]

    min_h = packages.iwal.iwal_functions._get_min_hypothesis(hypothesis_space, s, log_loss, labels_t)
    if min_h is lr:
        assert True
    else:
        assert False


def test__sum_losses():
    # example data set
    X = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
    y = [1, 0]
    lr = LogisticRegression().fit(X, y)

    # example labeled set
    x_t = np.asarray([3, 1]).reshape(1, -1)  # single element with 2 features
    y_t = np.asarray([1])
    c_t = 0.1
    s = [(x_t, y_t, c_t)]

    labels_t = [0, 1]

    # define dummy loss function for testing purposes
    def loss_func(a, b, labels):
        return a

    total = packages.iwal.iwal_functions._sum_losses(lr, s, loss_func, labels_t)
    assert total == 0.1


def test_iwal_query_selected_for_labeling():
    # example data set
    X = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
    y = [1, 0]
    lr = LogisticRegression().fit(X, y)
    hypothesis_space = [lr]

    # example labeled set
    x_t = np.asarray([3, 1]).reshape(1, -1)  # single element with 2 features
    y_t = np.asarray([1])
    selected = []

    # dummy function for testing
    def rejection_func(x_t, history):
        return 1.0

    # define dummy loss function for testing purposes
    def loss_func(a, b, labels):
        l = labels
        return a

    history = {
        'X': [],
        'y': [],
        'c': [],
        'Q': []
    }

    labels_t = [0, 1]

    h_t = packages.iwal.iwal_functions.iwal_query(x_t, y_t, selected, rejection_func, history, hypothesis_space,
                                                  loss_func, labels_t)

    assert len(selected) == 1
    if h_t is lr:
        assert True
    else:
        assert False


def test_iwal_query_not_selected_for_labeling():
    # example data set
    X = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
    y = [1, 0]
    lr = LogisticRegression().fit(X, y)
    hypothesis_space = [lr]

    # example labeled set
    x_t = np.asarray([3, 1]).reshape(1, -1)  # single element with 2 features
    y_t = np.asarray([1])
    selected = []

    # dummy function for testing
    def rejection_func(x_t, history):
        return 0.0

    # define dummy loss function for testing purposes
    def loss_func(a, b, labels):
        l = labels
        return a + b

    history = {
        'X': [],
        'y': [],
        'c': [],
        'Q': []
    }

    labels_t = [0, 1]

    h_t = packages.iwal.iwal_functions.iwal_query(x_t, y_t, selected, rejection_func, history, hypothesis_space,
                                                  loss_func, labels_t)

    assert len(selected) == 0
    if h_t is lr:
        assert True
    else:
        assert False


def test__loss_difference():
    test_y_true = np.asarray([0, 1])
    test_y_pred_i = np.asarray([1, 1])
    test_y_pred_j = np.asarray([0, 1])
    labels = [0, 1]

    actual = packages.iwal.iwal_functions._loss_difference(test_y_true, test_y_pred_i, test_y_pred_j, hinge_loss,
                                                           labels)
    assert actual == 0.5


def test__bootstrap_probability():
    p_min = 0.1
    max_loss_difference = 0.5
    actual = packages.iwal.iwal_functions._bootstrap_probability(p_min, max_loss_difference)
    assert actual == 0.55


def test_bootstrap_one_hypothesis():
    x = [[3, 1]]

    # example data set
    X = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
    y = [1, 0]
    lr = LogisticRegression().fit(X, y)
    hypothesis_space = [lr]
    p_min = 0.1
    labels = [0, 1]
    history = dict()
    additional = {
        'H': hypothesis_space,
        'loss': hinge_loss,
        'labels': labels,
        'p_min': p_min
    }

    actual = packages.iwal.iwal_functions.bootstrap(x, history, additional)
    assert actual == p_min
