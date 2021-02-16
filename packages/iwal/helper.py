from sklearn.metrics import hinge_loss
import numpy as np


# done, testable
def calculate_hinge_loss(x, h, label, labels):
    y_true = np.full(shape=len(x), fill_value=label, dtype=np.int)
    pred_decision = h.decision_function(x)
    loss = hinge_loss(y_true, pred_decision, labels=labels)

    return loss
