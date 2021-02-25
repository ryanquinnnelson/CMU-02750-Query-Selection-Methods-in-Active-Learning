"""
Unit tests for iwal module in the iwal package.
"""

import packages.iwal.iwal as iw
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError


def test__append_history_no_history():
    history = {}

    x_t = np.array([1, 2])
    y_t = np.array([1])
    p_t = 0.5
    q_t = 1

    iw._append_history(history, x_t, y_t, p_t, q_t)

    assert x_t in history['X']
    assert y_t in history['y']
    assert p_t in history['c']
    assert history['X'].shape == (1, 2)
    assert history['y'].shape == (1,)
    assert q_t in history['Q']


def test__append_history_history_exists():
    history = {
        'X': np.array([[1, 3]]),
        'y': np.array([0]),
        'c': [0.4],
        'Q': [0]
    }

    x_t = np.array([1, 2])
    y_t = np.array([1])
    p_t = 0.5
    q_t = 1

    iw._append_history(history, x_t, y_t, p_t, q_t)

    assert x_t in history['X']
    assert y_t in history['y']
    assert p_t in history['c']
    assert q_t in history['Q']
    assert history['X'].shape == (2, 2)
    assert history['y'].shape == (2,)


def test__add_to_selected_no_selected():
    selected = {}

    x_t = np.array([1, 2])
    y_t = np.array([1])
    c_t = 0.5

    iw._add_to_selected(selected, x_t, y_t, c_t)

    assert x_t in selected['X']
    assert selected['X'].shape == (1, 2)
    assert y_t in selected['y']
    assert selected['y'].shape == (1,)
    assert c_t in selected['c']


def test__add_to_selected_selected_exists():
    selected = {
        'X': np.array([[1, 3]]),
        'y': np.array([0]),
        'c': [0.4],
    }

    x_t = np.array([1, 2])
    y_t = np.array([1])
    c_t = 0.5

    iw._add_to_selected(selected, x_t, y_t, c_t)

    assert x_t in selected['X']
    assert y_t in selected['y']
    assert c_t in selected['c']


def test__add_to_selected_selected_exists_large_count():
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

    selected = {
        'X': X_train,
        'y': y_train,
        'c': [1.0 for _ in range(10)]
    }

    x_t = np.array([1, 2])
    y_t = np.array([1])
    c_t = 0.5

    iw._add_to_selected(selected, x_t, y_t, c_t)

    assert x_t in selected['X']
    assert y_t in selected['y']
    assert c_t in selected['c']


def test__choose_flip_action_heads():
    s = {}
    x_t = np.asarray([1, 2])
    y_t = np.asarray([1])
    p_t = 0.25
    Q_t = 1
    iw._choose_flip_action(Q_t, s, x_t, y_t, p_t)
    assert len(s['X']) == 1
    c_t = s['c'][0]
    assert c_t == 1 / p_t


def test__choose_flip_action_tails():
    s = {}
    x_t = np.asarray([1, 2])
    y_t = np.asarray([1])
    p_t = 0.25
    Q_t = 0
    iw._choose_flip_action(Q_t, s, x_t, y_t, p_t)
    assert 'X' not in s


def test__all_labels_in_selected_success():
    selected = {'y': np.array([0, 1])}
    labels = [0, 1]
    assert iw._all_labels_in_selected(selected, labels)


def test__all_labels_in_selected_failure():
    selected = {'y': np.array([np.array([0])])}
    labels = [0, 1]
    assert iw._all_labels_in_selected(selected, labels) is False


def test_iwal_query_bootstrap_selected_not_all_labels():
    history = {}
    selected = {}
    x_t = np.array([3, 1])
    y_t = np.array([1])

    def loss_function(h, x_t, label, labels):
        y_pred = h.predict(x_t)[0]
        return label * y_pred

    actual = iw.iwal_query(x_t, y_t, history, selected, 'bootstrap', loss_function)
    assert len(history['X']) == 1
    assert len(history['y']) == 1
    assert len(history['c']) == 1
    assert len(history['Q']) == 1
    assert len(selected['X']) == 1
    assert len(selected['y']) == 1
    assert len(selected['c']) == 1
    assert actual is None


def test_iwal_query_unimplemented_rejection_threshold():
    history = {}
    selected = {}
    x_t = np.array([3, 1])
    y_t = np.array([1])

    def loss_function(h, x_t, label, labels):
        y_pred = h.predict(x_t)[0]
        return label * y_pred

    with pytest.raises(NotImplementedError):
        iw.iwal_query(x_t, y_t, history, selected, 'other', loss_function)


def test_iwal_query_bootstrap_selected_all_labels():
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

    selected = {
        'X': X_train,
        'y': y_train,
        'c': [1.0 for _ in range(10)]
    }

    def loss_function(h, x_t, label, labels):
        y_pred = h.predict(x_t)[0]
        return label * y_pred

    history = {}
    x_t = np.array([3, 1])
    y_t = np.array([1])

    h_t = iw.iwal_query(x_t, y_t, history, selected, 'bootstrap', loss_function)

    # test that model has been fitted
    try:
        h_t.predict([[3, 1]])
    except NotFittedError:
        assert False, 'Model has not been fitted.'
