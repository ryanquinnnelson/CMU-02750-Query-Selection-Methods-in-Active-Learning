import packages.iwal.iwal_functions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hinge_loss, log_loss


def test_append_history_dictionary_matches():
    h = {
        'X': [],
        'y': [],
        'c': [],
        'Q': []
    }

    x_t = [1, 2]
    y_t = 1
    p_t = 0.5
    q_t = 1

    packages.iwal.iwal_functions._append_history(h, x_t, y_t, p_t, q_t)

    assert x_t in h['X']
    assert y_t in h['y']
    assert p_t in h['c']
    assert q_t in h['Q']


def test__get_min_hypothesis_empty_hypothesis_space():
    h = []
    s = {(1, 1, 0.5)}
    labels = [0, 1]

    min_h = packages.iwal.iwal_functions._get_min_hypothesis(h, s, log_loss, labels)
    if min_h is None:
        assert True
    else:
        assert False


def test__get_min_hypothesis_empty_set():
    lr = LogisticRegression()
    h = [lr]
    s = set()
    labels = [0, 1]

    min_h = packages.iwal.iwal_functions._get_min_hypothesis(h, s, log_loss, labels)
    if min_h is lr:
        assert True
    else:
        assert False


