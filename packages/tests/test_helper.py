# from sklearn.linear_model import LogisticRegression
# import packages.iwal.helper
# from sklearn.metrics import hinge_loss, log_loss
#
#
# def test_calculate_hinge_loss():
#     X_train = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
#     y_train = [1, 0]
#     h = LogisticRegression().fit(X_train, y_train)
#
#     # new sample
#     x = [[3, 1]]
#     y_true = [1]
#     labels = [0, 1]
#     pred_decision = h.decision_function(x)
#     loss = hinge_loss(y_true, pred_decision, labels=labels)
#
#     assert loss == packages.iwal.helper.calculate_hinge_loss(x, h, y_true, labels)
#
#
# def test_calculate_log_loss():
#     X_train = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
#     y_train = [1, 0]
#     h = LogisticRegression().fit(X_train, y_train)
#
#     # new sample
#     x = [[3, 1]]
#     y_true = [1]
#     labels = [0, 1]
#     y_pred = h.predict_proba(x)
#     loss = log_loss(y_true, y_pred, labels=labels)
#
#     assert loss == packages.iwal.helper.calculate_log_loss(x, h, y_true, labels)
