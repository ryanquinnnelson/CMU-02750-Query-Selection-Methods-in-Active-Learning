"""
Unit tests for loss_function module in the iwal package.
"""

from sklearn.linear_model import LogisticRegression
import packages.iwal.loss_function as lf
import numpy as np


def test_normalized_hinge_loss_1_first_binary_label():
    # prepare predictor
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

    lr = LogisticRegression().fit(X_train, y_train)

    # test value
    x = np.array([3, 1])
    y = 0

    # labels
    labels = [0, 1]

    expected = 1.0
    actual = lf.normalized_hinge_loss_1(lr, x, y, labels)
    assert actual == expected


def test_normalized_hinge_loss_2_first_binary_label():
    # prepare predictor
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

    lr = LogisticRegression().fit(X_train, y_train)

    # test value
    x = np.array([3, 1])
    y = 0

    # labels
    labels = [0, 1]

    expected = 1.9413497666501573 / 2.0
    actual = lf.normalized_hinge_loss_2(lr, x, y, labels)
    assert actual == expected


def test_normalized_hinge_loss_3_first_binary_label():
    # prepare predictor
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

    lr = LogisticRegression().fit(X_train, y_train)

    # test value
    x = np.array([3, 1])
    y = 0

    # labels
    labels = [0, 1]

    expected = 3.7757232135061716 / 10.0
    actual = lf.normalized_hinge_loss_3(lr, x, y, labels)
    assert actual == expected
