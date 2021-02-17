# import packages.iwal.iwal as iw
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import hinge_loss
# import numpy as np
# import pytest
#
#
# def test_append_history_dictionary_matches():
#     history = {
#         'X': [],
#         'y': [],
#         'c': [],
#         'Q': []
#     }
#
#     x_t = [1, 2]
#     y_t = 1
#     p_t = 0.5
#     q_t = 1
#
#     iw._append_history(history, x_t, y_t, p_t, q_t)
#
#     assert x_t in history['X']
#     assert y_t in history['y']
#     assert p_t in history['c']
#     assert q_t in history['Q']
#
#
# def test__choose_flip_action_heads():
#     s = []
#     x_t = np.asarray([1, 2, 3])
#     y_t = np.asarray([1])
#     p_t = 0.25
#     p_min = 0.1
#     Q_t = 1
#     iw._choose_flip_action(Q_t, s, x_t, y_t, p_t, p_min)
#     assert len(s) == 1
#     c_t = s[0][2]
#     assert c_t == p_min / p_t
#
#
# def test__choose_flip_action_tails():
#     s = []
#     x_t = np.asarray([1, 2, 3])
#     y_t = np.asarray([1])
#     p_t = 0.25
#     p_min = 0.1
#     Q_t = 0
#     iw._choose_flip_action(Q_t, s, x_t, y_t, p_t, p_min)
#     assert len(s) == 0
#
#
# def test__loss_summation_function():
#     # example data set
#     X = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
#     y = [1, 0]
#     lr = LogisticRegression().fit(X, y)
#
#     # example labeled set
#     s = [([[3, 1]], [1], 0.1),
#          ([[4, 1]], [0], 0.2), ]
#     labels = [0, 1]
#
#     # calculate expected loss
#     expected = 0.0
#     for x, y, c in s:
#         df = lr.decision_function(x)
#         expected += c * hinge_loss(y, df, labels=labels)
#
#     actual = iw._loss_summation_function(lr, s, labels)
#     assert actual == expected
#
#
# def test__get_min_hypothesis():
#     """
#     Testing whether minimization search works, not the specifics of
#     the loss_summation_function.
#     :return:
#     """
#
#     h_space = [1, 2, 3]
#     selected = [4, 5, 6]
#     labels = [0, 1]
#
#     def test_loss_summation_function(h, selected, label_list):
#         total = 0.0
#         for s in selected:
#             total += h * s
#         return total
#
#     min_h = iw._get_min_hypothesis(h_space, selected, labels, test_loss_summation_function)
#     assert min_h == 1
#
#
# def test_iwal_query_notimplemented():
#     history = {'X': [], 'y': [], 'c': [], 'Q': []}
#     with pytest.raises(NotImplementedError):
#         iw.iwal_query([[1]], [1], [], history, [], [], 'other', 1)
#
#
# def test_iwal_query_bootstrap_t_one_less_than_bootstrap_size():
#     history = {'X': [], 'y': [], 'c': [], 'Q': []}
#     bootstrap_size = 1
#     actual = iw.iwal_query([[3, 1]], [1], [], history, [], [], 'bootstrap', bootstrap_size)
#     assert len(history['X']) == 1
#     assert len(history['y']) == 1
#     assert len(history['c']) == 1
#     assert len(history['Q']) == 1
#     assert actual is None
