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
    x_t = [[3, 1]]
    y_t = [1]
    c_t = 0.1
    s = [(x_t, y_t, c_t)]

    labels_t = [0, 1]

    # define dummy loss function for testing purposes
    def loss_func(a, b, labels):
        l = labels
        return a + b

    total = packages.iwal.iwal_functions._sum_losses(lr, s, loss_func, labels_t)
    assert total == 0.2
