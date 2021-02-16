import packages.iwal.rejection_threshold as rt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hinge_loss
from sklearn.exceptions import NotFittedError
import pytest
import numpy as np


def test__bootstrap_calculate_p_t():
    p_min = 0.1
    max_loss_difference = 0.5
    actual = rt._bootstrap_calculate_p_t(p_min, max_loss_difference)
    assert actual == 0.55


def test__bootstrap_calculate_max_loss_difference():
    """
    Tests the maximization functionality only, not the loss_difference_function.
    :return:
    """

    x = 1
    h_space = [2, 3, 4]
    labels = [0, 1]

    def test_loss_difference_function(x, h_i, h_j, label, label_list):
        return (x * h_i) - (x * h_j) + label

    expected = 3
    actual = rt._bootstrap_calculate_max_loss_difference(x, h_space, labels, test_loss_difference_function)
    assert actual == expected


def test__bootstrap_ldf_hinge():
    # example data set
    X1 = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
    y1 = [1, 0]
    lr1 = LogisticRegression().fit(X1, y1)

    X2 = [[0, 0], [10, 10]]
    y2 = [1, 0]
    lr2 = LogisticRegression().fit(X2, y2)

    # example labeled set
    labels = [0, 1]

    x3 = [[3, 1]]
    y3 = [1]

    df1 = lr1.decision_function(x3)
    df2 = lr2.decision_function(x3)

    hl1 = hinge_loss(y3, df1, labels=labels)
    hl2 = hinge_loss(y3, df2, labels=labels)
    expected = hl1 - hl2

    actual = rt._bootstrap_ldf_hinge(x3, lr1, lr2, y3, labels)
    assert actual == expected


def test__bootstrap_reshape_history():
    x1 = np.asarray([[2.59193175, 1.14706863]])
    x2 = np.asarray([[1.7756532, 1.15670278]])
    X_before = [x1, x2]

    y1 = np.asarray([1])
    y2 = np.asarray([0])
    y_before = [y1, y2]
    history = {'X': X_before, 'y': y_before}
    X_after, y_after = rt._bootstrap_reshape_history(history)

    assert X_after.shape == (2, 2)
    assert y_after.shape == (2,)


def test__bootstrap_train_predictors():
    """
    Method used to check if model is fitted was sourced from:
    https://stackoverflow.com/questions/39884009/whats-the-best-way-to-test-whether-an-sklearn-model-has-been-fitted/51200847
    :return:
    """

    h_space = []
    for i in range(2):
        lr = LogisticRegression()
        h_space.append(lr)

    x1 = np.asarray([[2.59193175, 1.14706863]])
    x2 = np.asarray([[1.7756532, 1.15670278]])
    X_before = [x1, x2]

    y1 = np.asarray([1])
    y2 = np.asarray([0])
    y_before = [y1, y2]
    history = {'X': X_before, 'y': y_before}

    rt._bootstrap_train_predictors(h_space, history)

    x = [[3, 1]]
    for model in h_space:
        try:
            model.predict(x)
        except NotFittedError:
            assert False, 'Model has not been fitted.'


def test_bootstrap_t_less_than_bootstrap_size():
    history = {'X': []}
    b = 2

    h_space = []
    for i in range(2):
        lr = LogisticRegression()
        h_space.append(lr)

    expected = 1.0
    actual = rt.bootstrap(1, h_space, b, history, [0, 1])
    assert actual == expected

    # should raise error because models should not be trained
    with pytest.raises(NotFittedError):
        for model in h_space:
            model.predict([[3, 1]])

def test_bootstrap_t_equals_bootstrap_size():
    x1 = np.asarray([[2.59193175, 1.14706863]])
    x2 = np.asarray([[1.7756532, 1.15670278]])
    X_before = [x1, x2]

    y1 = np.asarray([1])
    y2 = np.asarray([0])
    y_before = [y1, y2]
    history = {'X': X_before, 'y': y_before}
    b = 2

    h_space = []
    for i in range(2):
        lr = LogisticRegression()
        h_space.append(lr)

    expected = 1.0
    actual = rt.bootstrap(1, h_space, b, history, [0, 1])
    assert actual == expected

    for model in h_space:
        try:
            model.predict([[3, 1]])
        except NotFittedError:
            assert False, 'Model has not been fitted.'
