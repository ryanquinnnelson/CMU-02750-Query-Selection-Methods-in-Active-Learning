# import packages.iwal.iwal_functions
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import hinge_loss, log_loss
# import numpy as np
# import pytest
# from sklearn import svm
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
#     packages.iwal.iwal_functions._append_history(history, x_t, y_t, p_t, q_t)
#
#     assert x_t in history['X']
#     assert y_t in history['y']
#     assert p_t in history['c']
#     assert q_t in history['Q']
#
#
# def test__loss_difference_hinge_loss_for_multiple_elements():
#
#     X = [[0], [1]]
#     y = [-1, 1]
#     est1 = svm.LinearSVC(random_state=0)
#     est1.fit(X, y)
#     pred_decision1 = est1.decision_function([[-2], [3], [0.5]])
#     loss1 = hinge_loss([-1, 1, 1], pred_decision1)
#
#     est2 = LogisticRegression()
#     est2.fit(X, y)
#     pred_decision2 = est2.decision_function([[-2], [3], [0.5]])
#     loss2 = hinge_loss([-1, 1, 1], pred_decision2)
#
#     test_y_true = np.asarray([-1, 1, 1])
#     labels = [0, 1]
#
#     expected = loss1 - loss2
#
#     actual = packages.iwal.iwal_functions._loss_difference_hinge_loss(test_y_true,pred_decision1,pred_decision2,labels)
#     assert actual == expected
#
#
# def test__loss_difference_for_not_implemented_loss_function():
#
#     X = [[0], [1]]
#     y = [-1, 1]
#     est1 = svm.LinearSVC(random_state=0)
#     est1.fit(X, y)
#     pred_decision1 = est1.decision_function([[-2], [3], [0.5]])
#
#     est2 = LogisticRegression()
#     est2.fit(X, y)
#     pred_decision2 = est2.decision_function([[-2], [3], [0.5]])
#
#     test_y_true = np.asarray([-1, 1, 1])
#     labels = [0, 1]
#     loss_function = 'log_loss'
#
#     with pytest.raises(NotImplementedError):
#         packages.iwal.iwal_functions._loss_difference(test_y_true, pred_decision1, pred_decision2,
#                                                                       labels, loss_function)
#
#
# def test__loss_difference_for_hinge_loss_different_models():
#
#     X = [[0], [1]]
#     y = [-1, 1]
#     est1 = svm.LinearSVC(random_state=0)
#     est1.fit(X, y)
#     pred_decision1 = est1.decision_function([[-2], [3], [0.5]])
#     loss1 = hinge_loss([-1, 1, 1], pred_decision1)
#
#     est2 = LogisticRegression()
#     est2.fit(X, y)
#     pred_decision2 = est2.decision_function([[-2], [3], [0.5]])
#     loss2 = hinge_loss([-1, 1, 1], pred_decision2)
#
#     test_y_true = np.asarray([-1, 1, 1])
#     labels = [0, 1]
#     loss_function = 'hinge_loss'
#
#     expected = loss1 - loss2
#
#     actual = packages.iwal.iwal_functions._loss_difference(test_y_true, pred_decision1, pred_decision2,
#                                                                       labels, loss_function)
#     assert actual == expected
#
#
# def test__loss_difference_for_hinge_loss_same_models():
#     # example data set
#     X1 = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
#     y1 = [1, 0]
#     lr1 = LogisticRegression().fit(X1, y1)
#
#     X2 = [[0, 0], [10, 10]]
#     y2 = [1, 0]
#     lr2 = LogisticRegression().fit(X2, y2)
#
#     # example labeled set
#     labels = [0, 1]
#
#     x3 = [[3, 1]]
#     y3 = [1]
#
#     df1 = lr1.decision_function(x3)
#     df2 = lr2.decision_function(x3)
#
#     hl1 = hinge_loss(y3, df1, labels=labels)
#     hl2 = hinge_loss(y3, df2, labels=labels)
#     expected = hl1 - hl2
#     loss_function = 'hinge_loss'
#
#     actual = packages.iwal.iwal_functions._loss_difference(y3, df1, df2,labels, loss_function)
#     assert actual == expected
#
#
# def test__bootstrap_probability():
#     p_min = 0.1
#     max_loss_difference = 0.5
#     actual = packages.iwal.iwal_functions._bootstrap_probability(p_min, max_loss_difference)
#     assert actual == 0.55
#
#
# def test__get_prediction_hinge_loss():
#
#     X = [[0], [1]]
#     y = [-1, 1]
#     lr = LogisticRegression()
#     lr.fit(X, y)
#     x = [[-2]]
#     pred_decision = lr.decision_function(x)
#
#     assert pred_decision == packages.iwal.iwal_functions._get_prediction(x, lr, 'hinge_loss')
#
#
# def test__get_prediction_log_loss():
#
#     X = [[0], [1]]
#     y = [-1, 1]
#     lr = LogisticRegression()
#     lr.fit(X, y)
#     x = [[-2]]
#     expected = lr.predict_proba(x).tolist()
#     actual = packages.iwal.iwal_functions._get_prediction(x, lr, 'log_loss').tolist()
#
#     assert expected == actual
#
#
# def test__get_prediction_not_implemented():
#
#     X = [[0], [1]]
#     y = [-1, 1]
#     lr = LogisticRegression()
#     lr.fit(X, y)
#     x = [[-2]]
#
#     with pytest.raises(NotImplementedError):
#         packages.iwal.iwal_functions._get_prediction(x, lr, 'other')
#
#
# def test__calculate_loss_hinge_loss():
#     X = [[0], [1]]
#     y = [-1, 1]
#     est = svm.LinearSVC(random_state=0)
#     est.fit(X, y)
#     pred_decision = est.decision_function([[-2], [3], [0.5]])
#     y_true = [-1, 1, 1]
#     expected = hinge_loss(y_true, pred_decision)
#     labels = [-1,1]
#     loss_function = 'hinge_loss'
#
#     actual = packages.iwal.iwal_functions._calculate_loss(y_true, pred_decision,labels,loss_function)
#     assert expected == actual
#
#
# def test__sum_losses_hinge_loss():
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
#     expected = 0.0
#     for x, y, c in s:
#         df = lr.decision_function(x)
#         expected += c * hinge_loss(y, df, labels=labels)
#
#     loss_function = 'hinge_loss'
#     actual = packages.iwal.iwal_functions._sum_losses(lr, s, labels, loss_function)
#     assert actual == expected
#
#
# def test__get_min_hypothesis():
#     # example data set
#     X1 = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
#     y1 = [1, 0]
#     lr1 = LogisticRegression().fit(X1, y1) #sum hinge loss is 0.4678294064768459
#
#     X2 = [[0, 0], [10, 10]]
#     y2 = [1, 0]
#     lr2 = LogisticRegression().fit(X2, y2) #sum hinge loss is 0.8374835459783136
#
#     H = [lr1,lr2]
#
#     # example labeled set
#     s = [([[3, 1]], [1], 0.1),
#          ([[4, 1]], [0], 0.2), ]
#     labels = [0, 1]
#     loss_function = 'hinge_loss'
#
#     expected = lr1
#     actual = packages.iwal.iwal_functions._get_min_hypothesis(H, s, labels, loss_function )
#     assert expected == actual
#
#
# def test__choose_flip_action_heads():
#     s = []
#     x_t = np.asarray([1,2,3])
#     y_t = np.asarray([1])
#     p_t = 0.25
#     Q_t = 1
#     packages.iwal.iwal_functions._choose_flip_action(Q_t, s, x_t, y_t, p_t)
#     assert len(s) ==1
#     assert s[0][2] == 1/0.25
#
#
# def test__choose_flip_action_tails():
#     s = []
#     x_t = np.asarray([1, 2, 3])
#     y_t = np.asarray([1])
#     p_t = 0.25
#     Q_t = 0
#     packages.iwal.iwal_functions._choose_flip_action(Q_t, s, x_t, y_t, p_t)
#     assert len(s) == 0
#
#
#
#
#
#     #
#     #
#     # def test_iwal_query_selected_for_labeling():
#     #     # example data set
#     #     X = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
#     #     y = [1, 0]
#     #     lr = LogisticRegression().fit(X, y)
#     #     hypothesis_space = [lr]
#     #
#     #     # example labeled set
#     #     x_t = np.asarray([3, 1]).reshape(1, -1)  # single element with 2 features
#     #     y_t = np.asarray([1])
#     #     selected = []
#     #
#     #     # dummy function for testing
#     #     def rejection_func(x_t, history):
#     #         return 1.0
#     #
#     #     # define dummy loss function for testing purposes
#     #     def loss_func(a, b, labels):
#     #         l = labels
#     #         return a
#     #
#     #     history = {
#     #         'X': [],
#     #         'y': [],
#     #         'c': [],
#     #         'Q': []
#     #     }
#     #
#     #     labels_t = [0, 1]
#     #
#     #     h_t = packages.iwal.iwal_functions.iwal_query(x_t, y_t, selected, rejection_func, history, hypothesis_space,
#     #                                                   loss_func, labels_t)
#     #
#     #     assert len(selected) == 1
#     #     if h_t is lr:
#     #         assert True
#     #     else:
#     #         assert False
#     #
#     #
#     # def test_iwal_query_not_selected_for_labeling():
#     #     # example data set
#     #     X = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
#     #     y = [1, 0]
#     #     lr = LogisticRegression().fit(X, y)
#     #     hypothesis_space = [lr]
#     #
#     #     # example labeled set
#     #     x_t = np.asarray([3, 1]).reshape(1, -1)  # single element with 2 features
#     #     y_t = np.asarray([1])
#     #     selected = []
#     #
#     #     # dummy function for testing
#     #     def rejection_func(x_t, history):
#     #         return 0.0
#     #
#     #     # define dummy loss function for testing purposes
#     #     def loss_func(a, b, labels):
#     #         l = labels
#     #         return a + b
#     #
#     #     history = {
#     #         'X': [],
#     #         'y': [],
#     #         'c': [],
#     #         'Q': []
#     #     }
#     #
#     #     labels_t = [0, 1]
#     #
#     #     h_t = packages.iwal.iwal_functions.iwal_query(x_t, y_t, selected, rejection_func, history, hypothesis_space,
#     #                                                   loss_func, labels_t)
#     #
#     #     assert len(selected) == 0
#     #     if h_t is lr:
#     #         assert True
#     #     else:
#     #         assert False
#     #
#     #
#
#     # selected = [([[-2]],[-1],0.1),
#     #             ([[3]],[1],0.5),
#     #             ([[0.5]],[1],0.25)]
#
#
# def test__bootstrap_two_hypotheses():
#     # example data set
#     X1 = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
#     y1 = [1, 0]
#     lr1 = LogisticRegression().fit(X1, y1)
#
#     X2 = [[0, 0], [10, 10]]
#     y2 = [1, 0]
#     lr2 = LogisticRegression().fit(X2, y2)
#
#     H = [lr1, lr2]
#
#     # example labeled set
#     labels = [0, 1]
#     p_min = 0.1
#     loss_function = 'hinge_loss'
#
#     x3 = [[3, 1]]
#     y3 = [1]
#
#     df1 = lr1.decision_function(x3)
#     df2 = lr2.decision_function(x3)
#
#     hl1 = hinge_loss(y3, df1, labels=labels)
#     hl2 = hinge_loss(y3, df2, labels=labels)
#     expected = p_min + (1 - p_min) * (hl2 - hl1)
#
#     actual = packages.iwal.iwal_functions._bootstrap(x3, H, labels, p_min,loss_function)
#     assert actual == expected
#
#
# #?? confirm this is the correct output once loss function is better known
# def test__iwal_query_bootstrap():
#
#     history = {
#         'X': [],
#         'y': [],
#         'c': [],
#         'Q': []
#     }
#
#
#     # example data set
#     X1 = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
#     y1 = [1, 0]
#     lr1 = LogisticRegression().fit(X1, y1)
#
#     X2 = [[0, 0], [10, 10]]
#     y2 = [1, 0]
#     lr2 = LogisticRegression().fit(X2, y2)
#
#     H = [lr1, lr2]
#     s = []
#
#     # example labeled set
#     labels = [0, 1]
#     p_min = 0.1
#     loss_function = 'hinge_loss'
#
#     x3 = [[3, 1]]
#     y3 = [1]
#
#     expected = lr1
#
#     actual = packages.iwal.iwal_functions.iwal_query_bootstrap(x3, y3, H, history, s, labels, loss_function, p_min)
#
#     assert actual == expected
