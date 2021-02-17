"""
Defines common functions shared throughout package.
"""
from sklearn.metrics import hinge_loss, log_loss
import numpy as np


# done, tested
def calculate_hinge_loss(x, h, label, labels):
    y_true = np.full(shape=len(x), fill_value=label, dtype=np.int)
    pred_decision = h.decision_function(x)
    loss = hinge_loss(y_true, pred_decision, labels=labels)

    return loss


# done, testable
def calculate_log_loss(x, h, label, labels):
    y_true = np.full(shape=len(x), fill_value=label, dtype=np.int)
    y_pred = h.predict_proba(x)
    loss = log_loss(y_true, y_pred, labels=labels)

    return loss
