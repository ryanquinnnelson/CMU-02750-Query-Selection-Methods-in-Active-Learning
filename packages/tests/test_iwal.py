from sklearn.exceptions import NotFittedError

import packages.iwal.iwal as iw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hinge_loss
import numpy as np
import pytest


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

    iw._append_history(history, x_t, y_t, p_t, q_t)

    assert x_t in history['X']
    assert y_t in history['y']
    assert p_t in history['c']
    assert q_t in history['Q']


def test__choose_flip_action_heads():
    s = {'X':[],'y':[],'c':[]}
    x_t = np.asarray([1, 2, 3])
    y_t = np.asarray([1])
    p_t = 0.25
    p_min = 0.1
    Q_t = 1
    iw._choose_flip_action(Q_t, s, x_t, y_t, p_t, p_min)
    assert len(s['X']) == 1
    c_t = s['c'][0]
    assert c_t == p_min / p_t


def test__choose_flip_action_tails():
    s = {'X':[],'y':[],'c':[]}
    x_t = np.asarray([1, 2, 3])
    y_t = np.asarray([1])
    p_t = 0.25
    p_min = 0.1
    Q_t = 0
    iw._choose_flip_action(Q_t, s, x_t, y_t, p_t, p_min)
    assert len(s['X']) == 0

def test__all_labels_in_selected_success():
    selected = {'y':[np.array([1]),np.array([0])]}
    labels = [0,1]
    assert iw._all_labels_in_selected(selected,labels)


def test__all_labels_in_selected_failure():
    selected = {'y':[np.array([0]),np.array([0])]}
    labels = [0,1]
    assert iw._all_labels_in_selected(selected,labels) is False



def test_iwal_query_bootstrap_selected_not_all_labels():
    history = {'X': [], 'y': [], 'c': [], 'Q': []}
    selected = {
        'X':[],
        'y':[],
        'c':[]
    }
    x_t = [[3, 1]]
    y_t= [1]

    actual = iw.iwal_query(x_t,y_t , history, selected,  'bootstrap')
    assert len(history['X']) == 1
    assert len(history['y']) == 1
    assert len(history['c']) == 1
    assert len(history['Q']) == 1
    assert len(selected['X']) ==1
    assert len(selected['y']) == 1
    assert len(selected['c']) == 1
    assert actual is None


# def test_iwal_query_bootstrap_selected_all_labels():
#     X = [[[2.59193175, 1.14706863]],
#          [[1.7756532, 1.15670278]],
#          [[2.8032241, 0.5802936]],
#          [[1.6090616, 0.61957339]],
#          [[2.04921553, 5.33233847]],
#          [[0.50554777, 4.05210011]],
#          [[1.07710058, 5.32177878]],
#          [[0.35482006, 2.9172298]],
#          [[1.96225112, 0.68921004]],
#          [[-0.16486876, 4.62773491]]]
#
#     y = [
#         [1],
#         [1],
#         [1],
#         [1],
#         [0],
#         [0],
#         [0],
#         [0],
#         [1],
#         [0]]
#
#
#     c = [1.0 for _ in range(10)]
#     #
#     #
#     # history = {'X': [], 'y': [], 'c': [], 'Q': []}
#     # selected = {
#     #     'X':X_train,
#     #     'y':y_train,
#     #     'c':[1.0 for each in range(10)]
#     # }
#     lr = LogisticRegression().fit(X,y,sample_weight=c)

    #
    # x_t = [3, 1]
    # y_t= 1
    # print(selected['X'])
    # print(selected['y'])
    # print(selected['c'])
    # h_t = iw.iwal_query(x_t,y_t , history, selected,  'bootstrap')

    # # test that model has been fitted
    # try:
    #     h_t.predict([[3, 1]])
    # except NotFittedError:
    #     assert False, 'Model has not been fitted.'
