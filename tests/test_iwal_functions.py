import packages.iwal.iwal_functions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hinge_loss, log_loss
import numpy as np
import pytest
from sklearn import svm


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

#
# def test__get_min_hypothesis_empty_hypothesis_space():
#     hypothesis_space = []
#     s = {(1, 1, 0.5)}
#     labels = [0, 1]
#
#     min_h = packages.iwal.iwal_functions._get_min_hypothesis(hypothesis_space, s, log_loss, labels)
#     if min_h is None:
#         assert True
#     else:
#         assert False
#
#
# def test__get_min_hypothesis_empty_set():
#     # example data set
#     X = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
#     y = [1, 0]
#
#     lr = LogisticRegression().fit(X, y)
#     hypothesis_space = [lr]
#     s = set()
#     labels = [0, 1]
#
#     min_h = packages.iwal.iwal_functions._get_min_hypothesis(hypothesis_space, s, log_loss, labels)
#     if min_h is lr:
#         assert True
#     else:
#         assert False
#
#
# def test__get_min_hypothesis_one_hypothesis():
#     # example data set
#     X = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
#     y = [1, 0]
#     lr = LogisticRegression().fit(X, y)
#     hypothesis_space = [lr]
#
#     # example labeled set
#     x_t = [[3, 1]]
#     y_t = [1]
#     c_t = 0.1
#     s = [(x_t, y_t, c_t)]
#
#     labels_t = [0, 1]
#
#     min_h = packages.iwal.iwal_functions._get_min_hypothesis(hypothesis_space, s, log_loss, labels_t)
#     if min_h is lr:
#         assert True
#     else:
#         assert False
#
#
# def test__sum_losses():
#     # example data set
#     X = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
#     y = [1, 0]
#     lr = LogisticRegression().fit(X, y)
#
#     # example labeled set
#     x_t = np.asarray([3, 1]).reshape(1, -1)  # single element with 2 features
#     y_t = np.asarray([1])
#     c_t = 0.1
#     s = [(x_t, y_t, c_t)]
#
#     labels_t = [0, 1]
#
#     # define dummy loss function for testing purposes
#     def loss_func(a, b, labels):
#         return a
#
#     total = packages.iwal.iwal_functions._sum_losses(lr, s, loss_func, labels_t)
#     assert total == 0.1
#
#
# def test_iwal_query_selected_for_labeling():
#     # example data set
#     X = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
#     y = [1, 0]
#     lr = LogisticRegression().fit(X, y)
#     hypothesis_space = [lr]
#
#     # example labeled set
#     x_t = np.asarray([3, 1]).reshape(1, -1)  # single element with 2 features
#     y_t = np.asarray([1])
#     selected = []
#
#     # dummy function for testing
#     def rejection_func(x_t, history):
#         return 1.0
#
#     # define dummy loss function for testing purposes
#     def loss_func(a, b, labels):
#         l = labels
#         return a
#
#     history = {
#         'X': [],
#         'y': [],
#         'c': [],
#         'Q': []
#     }
#
#     labels_t = [0, 1]
#
#     h_t = packages.iwal.iwal_functions.iwal_query(x_t, y_t, selected, rejection_func, history, hypothesis_space,
#                                                   loss_func, labels_t)
#
#     assert len(selected) == 1
#     if h_t is lr:
#         assert True
#     else:
#         assert False
#
#
# def test_iwal_query_not_selected_for_labeling():
#     # example data set
#     X = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
#     y = [1, 0]
#     lr = LogisticRegression().fit(X, y)
#     hypothesis_space = [lr]
#
#     # example labeled set
#     x_t = np.asarray([3, 1]).reshape(1, -1)  # single element with 2 features
#     y_t = np.asarray([1])
#     selected = []
#
#     # dummy function for testing
#     def rejection_func(x_t, history):
#         return 0.0
#
#     # define dummy loss function for testing purposes
#     def loss_func(a, b, labels):
#         l = labels
#         return a + b
#
#     history = {
#         'X': [],
#         'y': [],
#         'c': [],
#         'Q': []
#     }
#
#     labels_t = [0, 1]
#
#     h_t = packages.iwal.iwal_functions.iwal_query(x_t, y_t, selected, rejection_func, history, hypothesis_space,
#                                                   loss_func, labels_t)
#
#     assert len(selected) == 0
#     if h_t is lr:
#         assert True
#     else:
#         assert False
#
#
def test__loss_difference_hinge_loss():

    X = [[0], [1]]
    y = [-1, 1]
    est1 = svm.LinearSVC(random_state=0)
    est1.fit(X, y)
    pred_decision1 = est1.decision_function([[-2], [3], [0.5]])
    loss1 = hinge_loss([-1, 1, 1], pred_decision1)

    est2 = LogisticRegression()
    est2.fit(X, y)
    pred_decision2 = est2.decision_function([[-2], [3], [0.5]])
    loss2 = hinge_loss([-1, 1, 1], pred_decision2)

    test_y_true = np.asarray([-1, 1, 1])
    labels = [0, 1]

    expected = loss1 - loss2

    actual = packages.iwal.iwal_functions._loss_difference_hinge_loss(test_y_true,pred_decision1,pred_decision2,labels)
    assert actual == expected


def test__loss_difference_implemented_loss_function():

    X = [[0], [1]]
    y = [-1, 1]
    est1 = svm.LinearSVC(random_state=0)
    est1.fit(X, y)
    pred_decision1 = est1.decision_function([[-2], [3], [0.5]])
    loss1 = hinge_loss([-1, 1, 1], pred_decision1)

    est2 = LogisticRegression()
    est2.fit(X, y)
    pred_decision2 = est2.decision_function([[-2], [3], [0.5]])
    loss2 = hinge_loss([-1, 1, 1], pred_decision2)

    test_y_true = np.asarray([-1, 1, 1])
    labels = [0, 1]
    loss_function = 'hinge_loss'

    expected = loss1 - loss2

    actual = packages.iwal.iwal_functions._loss_difference(test_y_true, pred_decision1, pred_decision2,
                                                                      labels, loss_function)
    assert actual == expected


def test__loss_difference_not_implemented_loss_function():

    X = [[0], [1]]
    y = [-1, 1]
    est1 = svm.LinearSVC(random_state=0)
    est1.fit(X, y)
    pred_decision1 = est1.decision_function([[-2], [3], [0.5]])

    est2 = LogisticRegression()
    est2.fit(X, y)
    pred_decision2 = est2.decision_function([[-2], [3], [0.5]])

    test_y_true = np.asarray([-1, 1, 1])
    labels = [0, 1]
    loss_function = 'log_loss'

    with pytest.raises(NotImplementedError):
        packages.iwal.iwal_functions._loss_difference(test_y_true, pred_decision1, pred_decision2,
                                                                      labels, loss_function)


def test__bootstrap_probability():
    p_min = 0.1
    max_loss_difference = 0.5
    actual = packages.iwal.iwal_functions._bootstrap_probability(p_min, max_loss_difference)
    assert actual == 0.55


def test__get_prediction_hinge_loss():

    X = [[0], [1]]
    y = [-1, 1]
    lr = LogisticRegression()
    lr.fit(X, y)
    x = [[-2]]
    pred_decision = lr.decision_function(x)

    assert pred_decision == packages.iwal.iwal_functions._get_prediction(x, lr, 'hinge_loss')


def test__get_prediction_log_loss():

    X = [[0], [1]]
    y = [-1, 1]
    lr = LogisticRegression()
    lr.fit(X, y)
    x = [[-2]]
    expected = lr.predict_proba(x).tolist()
    actual = packages.iwal.iwal_functions._get_prediction(x, lr, 'log_loss').tolist()

    assert expected == actual


def test__get_prediction_not_implemented():

    X = [[0], [1]]
    y = [-1, 1]
    lr = LogisticRegression()
    lr.fit(X, y)
    x = [[-2]]

    with pytest.raises(NotImplementedError):
        packages.iwal.iwal_functions._get_prediction(x, lr, 'other')


def test_bootstrap_two_hypotheses():
    X = [[0], [1]]
    y = [-1, 1]
    sv = svm.LinearSVC(random_state=0)
    sv.fit(X, y)

    lr = LogisticRegression()
    lr.fit(X, y)

    H = [sv,lr]
    x= [[-2]]
    labels = [0, 1]
    loss_function = 'hinge_loss'
    p_min = 0.1

    packages.iwal.iwal_functions._bootstrap(x, H, labels, p_min,loss_function)

