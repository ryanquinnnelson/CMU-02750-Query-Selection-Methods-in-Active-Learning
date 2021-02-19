import packages.iwal.rejection_threshold as rt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hinge_loss
from sklearn.exceptions import NotFittedError
import pytest
import numpy as np


def test__bootstrap_calc_z_value():
    x = 10
    min_value = 1
    max_value = 19
    expected = 0.5
    actual = rt._bootstrap_calc_z_value(x, min_value, max_value)
    assert actual == expected


def test__bootstrap_calc_z_value_zeros_only():
    x = 10
    min_value = 0.0
    max_value = 0.0
    expected = 0.0
    actual = rt._bootstrap_calc_z_value(x, min_value, max_value)
    assert actual == expected


def test__bootstrap_combine_p_min_and_max_loss():
    p_min = 0.1
    max_loss_difference = 0.5
    actual = rt._bootstrap_combine_p_min_and_max_loss(p_min, max_loss_difference)
    assert actual == 0.55


def test__bootstrap_combine_p_min_and_max_loss_less_than_zero():
    p_min = 0.1
    max_loss_difference = -0.1

    with pytest.raises(ValueError):
        rt._bootstrap_combine_p_min_and_max_loss(p_min, max_loss_difference)


def test__bootstrap_combine_p_min_and_max_loss_more_than_one():
    p_min = 0.1
    max_loss_difference = 1.1

    with pytest.raises(ValueError):
        rt._bootstrap_combine_p_min_and_max_loss(p_min, max_loss_difference)


def test__bootstrap_calc_max_loss_using_loss_function():
    """
    Tests the maximization functionality, not the loss_function.
    :return:
    """

    x = 1
    h_space = [.2, .3, .4]
    labels = [0, 1]

    def test_loss_function(x, h, label, label_list):
        return x * h

    expected = 0.2
    actual = rt._bootstrap_calc_max_loss_using_loss_function(x, h_space, labels, test_loss_function)
    assert actual == expected


def test__bootstrap_calc_normalized_max_loss_difference():
    min_loss = 1
    max_loss = 11
    loss_i = 10
    loss_j = 7
    expected = (loss_i - min_loss) / (max_loss - min_loss) - (loss_j - min_loss) / (max_loss - min_loss)
    actual = rt._bootstrap_calc_normalized_max_loss_difference(min_loss, max_loss, loss_i, loss_j)

    assert actual == expected


def test__bootstrap_get_min_max_loss_difference():
    # example data set
    X1 = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
    y1 = [1, 0]
    lr1 = LogisticRegression().fit(X1, y1)

    X2 = [[0, 0], [10, 10]]
    y2 = [1, 0]
    lr2 = LogisticRegression().fit(X2, y2)
    H = [lr1, lr2]

    # example labeled set
    labels = [0, 1]

    x3 = [[3, 1]]
    y3 = [1]

    df1 = lr1.decision_function(x3)
    df2 = lr2.decision_function(x3)

    hl1 = hinge_loss(y3, df1, labels=labels)  # min
    hl2 = hinge_loss(y3, df2, labels=labels)  # max

    min_loss, max_loss, max_diff_loss_i, max_diff_loss_j = rt._bootstrap_get_min_max_loss_difference(x3, H, labels)
    assert min_loss == hl1
    assert max_loss == hl2
    assert max_diff_loss_i == hl2
    assert max_diff_loss_j == hl1


def test__bootstrap_calc_max_loss_difference_internal():
    # example data set
    X1 = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
    y1 = [1, 0]
    lr1 = LogisticRegression().fit(X1, y1)

    X2 = [[0, 0], [10, 10]]
    y2 = [1, 0]
    lr2 = LogisticRegression().fit(X2, y2)
    H = [lr1, lr2]

    # example labeled set
    labels = [0, 1]

    x3 = [[3, 1]]
    y3 = [1]

    df1 = lr1.decision_function(x3)
    df2 = lr2.decision_function(x3)

    hl1 = hinge_loss(y3, df1, labels=labels)  # min
    hl2 = hinge_loss(y3, df2, labels=labels)  # max

    expected = 1.0
    actual = rt._bootstrap_calc_max_loss_difference_internal(x3, H, labels)
    assert actual == expected


def test__bootstrap_reshape_history_numpy_array():
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


def test__bootstrap_reshape_history_2d_list():
    x1 = [[2.59193175, 1.14706863]]
    x2 = [[1.7756532, 1.15670278]]
    X_before = [x1, x2]

    y1 = [1]
    y2 = [0]
    y_before = [y1, y2]
    history = {'X': X_before, 'y': y_before}
    X_after, y_after = rt._bootstrap_reshape_history(history)

    assert X_after.shape == (2, 2)
    assert y_after.shape == (2,)


# not currently testing i.i.d.
def test__bootstrap_select_iid_training_set():
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1])

    X_train, y_train = rt._bootstrap_select_iid_training_set(X, y)

    assert X_train.shape[0] == X.shape[0]
    assert y_train.shape[0] == y.shape[0]


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

    X_before = [[[2.59193175, 1.14706863]],
                [[1.7756532, 1.15670278]],
                [[2.8032241, 0.5802936]],
                [[1.6090616, 0.61957339]],
                [[2.04921553, 5.33233847]],
                [[0.50554777, 4.05210011]],
                [[1.07710058, 5.32177878]],
                [[0.35482006, 2.9172298]],
                [[1.96225112, 0.68921004]],
                [[-0.16486876, 4.62773491]]]

    y_before = [[1],
                [1],
                [1],
                [1],
                [0],
                [0],
                [0],
                [0],
                [1],
                [0]]
    history = {'X': X_before, 'y': y_before}

    rt._bootstrap_train_predictors(h_space, history)

    x = [[3, 1]]
    for model in h_space:
        try:
            model.predict(x)
        except NotFittedError:
            assert False, 'Model has not been fitted.'


def test_bootstrap_training_size_less_than_bootstrap_size():
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


def test_bootstrap_training_size_equals_bootstrap_size():
    X_before = [[[2.59193175, 1.14706863]],
                [[1.7756532, 1.15670278]],
                [[2.8032241, 0.5802936]],
                [[1.6090616, 0.61957339]],
                [[2.04921553, 5.33233847]],
                [[0.50554777, 4.05210011]],
                [[1.07710058, 5.32177878]],
                [[0.35482006, 2.9172298]],
                [[1.96225112, 0.68921004]],
                [[-0.16486876, 4.62773491]]]

    y_before = [[1],
                [1],
                [1],
                [1],
                [0],
                [0],
                [0],
                [0],
                [1],
                [0]]
    history = {'X': X_before, 'y': y_before}
    b = 10

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
