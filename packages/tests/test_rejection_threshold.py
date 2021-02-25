"""
Unit tests for rejection_threshold module in the iwal package.
"""

import packages.iwal.rejection_threshold as rt
from sklearn.exceptions import NotFittedError
import pytest
import numpy as np


def test__bootstrap_check_losses_success():
    loss_i = 0.0
    loss_j = 1.0
    rt._bootstrap_check_losses(loss_i, loss_j)


def test__bootstrap_check_losses_success_failure_i_low():
    loss_i = -0.1
    loss_j = 1.0

    with pytest.raises(ValueError):
        rt._bootstrap_check_losses(loss_i, loss_j)


def test__bootstrap_check_losses_success_failure_i_high():
    loss_i = 1.1
    loss_j = 1.0

    with pytest.raises(ValueError):
        rt._bootstrap_check_losses(loss_i, loss_j)


def test__bootstrap_check_losses_success_failure_j_low():
    loss_i = 1.0
    loss_j = -0.1

    with pytest.raises(ValueError):
        rt._bootstrap_check_losses(loss_i, loss_j)


def test__bootstrap_check_losses_success_failure_j_high():
    loss_i = 1.0
    loss_j = 1.1

    with pytest.raises(ValueError):
        rt._bootstrap_check_losses(loss_i, loss_j)


def test__bootstrap_calc_max_loss():
    predictors = [1, 2, 3, 4, 5]
    test_labels = [0, 1]
    x_t = .1

    def loss_function(h, x, label, labels):
        return h * x

    expected = .4
    actual = rt._bootstrap_calc_max_loss(x_t, predictors, test_labels, loss_function)
    assert actual == expected


def test__bootstrap_calc_max_loss_loss_difference_outside_bounds():
    predictors = [1, 2, 3, 4, 5]
    test_labels = [0, 1]
    x_t = .1

    def loss_function(h, x, label, labels):
        return 100 * x * h

    with pytest.raises(ValueError):
        rt._bootstrap_calc_max_loss(x_t, predictors, test_labels, loss_function)


def test__bootstrap_combine_p_min_and_max_loss():
    p_min = 0.1
    max_loss_difference = 0.5
    actual = rt._bootstrap_combine_p_min_and_max_loss(p_min, max_loss_difference)
    assert actual == 0.55


def test__bootstrap_y_has_all_labels_success():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
    labels = [0, 1]
    assert rt._bootstrap_y_has_all_labels(y, labels)


def test__bootstrap_y_has_all_labels_failure():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
    labels = [0, 1, 2]
    assert rt._bootstrap_y_has_all_labels(y, labels) is False


# test is not currently checking i.i.d. property
def test__bootstrap_select_iid_training_set():
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]])
    y = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
    labels = [0, 1]

    X_train, y_train = rt._bootstrap_select_iid_training_set(X, y, labels)

    assert len(X_train) == len(X)
    assert len(y_train) == len(y)
    for label in labels:
        assert label in y_train


def test__bootstrap_select_bootstrap_training_set():
    history = {
        'X': np.array([[1, 2], [2, 3], [3, 4], [4, 5]]),
        'y': np.array([0, 1, 1, 0])
    }
    bootstrap_size = 3
    X_expected = [[1, 2], [2, 3], [3, 4]]
    y_expected = [0, 1, 1]
    X_actual, y_actual = rt._bootstrap_select_bootstrap_training_set(history, bootstrap_size)
    assert X_actual.tolist() == X_expected
    assert y_actual.tolist() == y_expected


# not currently testing whether models are different
def test__bootstrap_train_predictors_success():
    """
    Method used to check if model is fitted was sourced from:
    https://stackoverflow.com/questions/39884009/whats-the-best-way-to-test-whether-an-sklearn-model-has-been-fitted/51200847
    :return:
    """
    X_train = np.array([[2.59193175, 1.14706863],
                        [1.7756532, 1.15670278],
                        [2.8032241, 0.5802936],
                        [1.6090616, 0.61957339],
                        [2.04921553, 5.33233847],
                        [0.50554777, 4.05210011],
                        [1.07710058, 5.32177878],
                        [0.35482006, 2.9172298],
                        [1.96225112, 0.68921004],
                        [-0.16486876, 4.62773491]])

    y_train = np.array([1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0])

    history = {'X': X_train, 'y': y_train}
    bootstrap_size = 10
    num_predictors = 3
    labels = [0, 1]

    predictors = rt._bootstrap_train_predictors(history, bootstrap_size, num_predictors, labels)
    for predictor in predictors:
        try:
            predictor.predict([[3, 1]])
        except NotFittedError:
            assert False, 'Model has not been fitted.'


def test__bootstrap_train_predictors_y_does_not_contain_all_labels():
    """
    Method used to check if model is fitted was sourced from:
    https://stackoverflow.com/questions/39884009/whats-the-best-way-to-test-whether-an-sklearn-model-has-been-fitted/51200847
    :return:
    """
    X_train = np.array([[2.59193175, 1.14706863],
                        [1.7756532, 1.15670278],
                        [2.8032241, 0.5802936],
                        [1.6090616, 0.61957339],
                        [2.04921553, 5.33233847],
                        [0.50554777, 4.05210011],
                        [1.07710058, 5.32177878],
                        [0.35482006, 2.9172298],
                        [1.96225112, 0.68921004],
                        [-0.16486876, 4.62773491]])

    y_train = np.array([1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1])

    history = {'X': X_train, 'y': y_train}
    bootstrap_size = 10
    num_predictors = 3
    labels = [0, 1]

    predictors = rt._bootstrap_train_predictors(history, bootstrap_size, num_predictors, labels)
    assert len(predictors) == 0


def test_bootstrap_history_less_than_bootstrap_size():
    x_t = [[3, 1]]
    history = {}
    loss_function = ''
    expected = 1.0
    actual = rt.bootstrap(x_t, history, loss_function)
    assert actual == expected


def test_bootstrap():
    X_train = np.array([[2.59193175, 1.14706863],
                        [1.7756532, 1.15670278],
                        [2.8032241, 0.5802936],
                        [1.6090616, 0.61957339],
                        [2.04921553, 5.33233847],
                        [0.50554777, 4.05210011],
                        [1.07710058, 5.32177878],
                        [0.35482006, 2.9172298],
                        [1.96225112, 0.68921004],
                        [-0.16486876, 4.62773491]])

    y_train = np.array([1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0])

    history = {'X': X_train, 'y': y_train}
    x_t = [[3, 1]]

    def loss_function(h, x_t, label, labels):
        y_pred = h.predict(x_t)[0]
        return label * y_pred

    expected = 0.1
    actual = rt.bootstrap(x_t, history, loss_function)
    assert actual == expected
