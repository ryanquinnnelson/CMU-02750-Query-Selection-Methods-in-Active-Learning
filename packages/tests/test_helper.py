from sklearn.linear_model import LogisticRegression
import packages.iwal.helper
from sklearn.metrics import hinge_loss, log_loss


# done, tested2
def _bootstrap_loss_function(predictor, x_t, y_true, labels):
    """

    :param predictor:
    :param x_t:
    :param y_true:
    :param labels:
    :return:
    """

    # get probability
    y_prob = predictor.predict_proba(x_t.reshape(1, -1))
    loss = log_loss([y_true], y_prob, labels=labels)
    return loss/10   # hard to get loss in correct range??


def test_calculate_hinge_loss():
    X_train = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
    y_train = [1, 0]
    h = LogisticRegression().fit(X_train, y_train)

    # new sample
    x = [[3, 1]]
    y_true = [1]
    labels = [0, 1]
    pred_decision = h.decision_function(x)
    loss = hinge_loss(y_true, pred_decision, labels=labels)

    assert loss == packages.iwal.helper.calculate_hinge_loss(x, h, y_true, labels)


def test_calculate_log_loss():
    X_train = [[2.59193175, 1.14706863], [1.7756532, 1.15670278]]
    y_train = [1, 0]
    h = LogisticRegression().fit(X_train, y_train)

    # new sample
    x = [[3, 1]]
    y_true = [1]
    labels = [0, 1]
    y_pred = h.predict_proba(x)
    loss = log_loss(y_true, y_pred, labels=labels)

    assert loss == packages.iwal.helper.calculate_log_loss(x, h, y_true, labels)


#
# def test__bootstrap_loss_function():
#     X_train = np.array([[2.59193175, 1.14706863],
#                         [1.7756532, 1.15670278],
#                         [2.8032241, 0.5802936],
#                         [1.6090616, 0.61957339],
#                         [2.04921553, 5.33233847],
#                         [0.50554777, 4.05210011],
#                         [1.07710058, 5.32177878],
#                         [0.35482006, 2.9172298],
#                         [1.96225112, 0.68921004],
#                         [-0.16486876, 4.62773491]])
#
#     y_train = np.array([1,
#                         1,
#                         1,
#                         1,
#                         0,
#                         0,
#                         0,
#                         0,
#                         1,
#                         0])
#
#     lr = LogisticRegression().fit(X_train, y_train)
#     x_t = np.array([3, 1])
#     y_true = np.array([1])
#     labels = [0, 1]
#     expected = log_loss(y_true, lr.predict_proba(x_t.reshape(1,-1)), labels=labels)
#     actual = rt._bootstrap_loss_function(lr, x_t, y_true, labels)
#     assert actual == expected
#
#