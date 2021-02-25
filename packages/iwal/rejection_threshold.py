"""
Implements rejection-threshold functions for the Importance Weighting Active Learning (IWAL) algorithm defined in
the paper by Beygelzimer et al. See https://arxiv.org/pdf/0812.4952.pdf.
"""
import numpy as np
from typing import Any, Tuple
from sklearn.linear_model import LogisticRegression


def _bootstrap_check_losses(loss_i: float, loss_j: float) -> None:
    """
    Checks whether given loss values are in [0,1]. Raises ValueError if values are outside this expected range.
    :param loss_i: Loss from predictor i.
    :param loss_j: Loss from predictor j.
    :return: None
    """
    if loss_i < 0.0 or loss_i > 1.0 or loss_j < 0.0 or loss_j > 1.0:
        raise ValueError('Loss must be within range [0,1]:', loss_i, loss_j)


def _bootstrap_calc_max_loss(x_t: np.ndarray, predictors: list, labels: list, loss_function: Any) -> float:
    """
    Calculates the maximum loss difference, considering every pair of predictors in the hypothesis space and every
    possible label.

    :param x_t: (2,) numpy array representing sample data point.
    :param predictors: List of predictors in the hypothesis space.
    :param labels: List of all possible labels for the data set.
    :param loss_function: Python function which calculates loss normalized to [0,1] range. The following signature is
                          expected: function_name(predictor:Any, x_t:np.ndarray, label:Any, labels:list) -> Float.
                          predictor: sklearn model;
                          x_t: (2,) numpy array representing sample data point;
                          label: single label from list of labels;
                          labels: List of all possible labels for the data set.
    :return: Float representing the maximum loss difference.
    """

    max_diff = -10000.0  # arbitrary starting value

    # consider every pair of models in the hypothesis space combined with every label
    for i in range(len(predictors)):
        for j in range(len(predictors)):
            for label in labels:

                h_i = predictors[i]
                h_j = predictors[j]

                # calculate loss difference
                loss_i = loss_function(h_i, x_t, label, labels)
                loss_j = loss_function(h_j, x_t, label, labels)
                _bootstrap_check_losses(loss_i, loss_j)
                diff = loss_i - loss_j

                # update max difference
                if diff > max_diff:
                    max_diff = diff

    return max_diff


def _bootstrap_combine_p_min_and_max_loss(p_min: float, max_loss_difference: float) -> float:
    """
    Performs final calculation for rejection threshold probability p_t using the following formula:

    p_t = p_min + (1 - p_min) * max_loss_difference
    :param p_min: Minimum probability for selecting a sample for labeling.
    :param max_loss_difference: Float representing the maximum loss difference considering all models and labels.
    :return: Float representing the rejection threshold probability. Value is in [0,1].
    """
    return p_min + (1 - p_min) * max_loss_difference


# to be tested
def _bootstrap_calculate_p_t(x_t: np.ndarray, predictors: list, labels: list, p_min: float,
                             loss_function: Any) -> float:
    """
    Calculates the rejection threshold probability p_t as defined by IWAL.
    Note: If list of predictors is empty, p_t is set to 1.0.

    :param x_t: (2,) numpy array representing sample data point.
    :param predictors: List of predictors in the hypothesis space.
    :param labels: List of all possible labels for the data set.
    :param p_min: Minimum probability for selecting a sample for labeling.
    :param loss_function: Python function which calculates loss normalized to [0,1] range. The following signature is
                          expected: function_name(predictor:Any, x_t:np.ndarray, label:Any, labels:list) -> Float.
                          predictor: sklearn model;
                          x_t: (2,) numpy array representing sample data point;
                          label: single label from list of labels;
                          labels: List of all possible labels for the data set.
    :return: Float representing the rejection threshold probability. Value is in [0,1].
    """

    if len(predictors) > 0:

        # calculate probability
        max_loss_diff = _bootstrap_calc_max_loss(x_t, predictors, labels, loss_function)
        p_t = _bootstrap_combine_p_min_and_max_loss(p_min, max_loss_diff)

    else:

        # default value is used
        p_t = 1.0

    return p_t


def _bootstrap_y_has_all_labels(y: np.ndarray, labels: list) -> bool:
    """
    Determines whether all labels are found in the array of selected labels.
    :param y_t: (n,) numpy array representing labels for n samples.
    :param labels: List of all possible labels for the data set.
    :return: True if all labels are found, False otherwise.
    """

    for label in labels:
        if label not in y:
            return False

    return True


def _bootstrap_select_iid_training_set(X: np.ndarray, y: np.ndarray, labels: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selects n random samples from the given data set, with replacement, where n is equal to the length of the data set.
    Note: If the selected training dataset is found to be missing labels, a new dataset is drawn from the same
    underlying distribution. This process is repeated until an i.i.d. training set with all labels is found.

    :param X: (n,2) numpy array representing data points for n samples.
    :param y_t: (n,) numpy array representing labels for n samples.
    :return: (np.ndarray, np.ndarray) Tuple containing (X_train,y_train), where X_train is a numpy array of sample
             features and y_train is a numpy array of labels.
    """

    n = len(X)  # size of desired training set

    # build a training set that contains all labels expected in the data set
    training_has_all_labels = False
    while not training_has_all_labels:
        indexes = np.random.choice(n, n, replace=True)  # select n indices from a range of 0 to n-1
        training_has_all_labels = _bootstrap_y_has_all_labels(y[indexes], labels)

    return X[indexes], y[indexes]


def _bootstrap_select_bootstrap_training_set(history: dict, bootstrap_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selects first b samples from query history, where b is the bootstrap_size.
    :param history: Dictionary containing query history. Expected to be empty or contain four keys: 'X','y','c','Q'.
                    'X' contains a numpy array (n,2) where each row represents a sample and n is the number of
                    queries made so far. 'y' contains a numpy array (n,) where each row represents a sample label.
                    'c' contains a list where each element represents the probability a sample was selected for
                    labeling. 'Q' contains a list where each element represents the 0-1 result of a coin flip.
    :param bootstrap_size: Number of samples to be used in bootstrapping.
    :return: (np.ndarray, np.ndarray) Tuple containing (X_train,y_train), where X_train is a numpy array of sample
             features and y_train is a numpy array of labels.
    """
    X_history = history['X'][:bootstrap_size]
    y_history = history['y'][:bootstrap_size]
    return X_history, y_history


def _bootstrap_train_predictors(history: dict, bootstrap_size: int, num_predictors: int, labels: list) -> list:
    """
    Trains p predictors to be used in the rejection threshold bootstrap algorithm, where p is the number of predictors.
    Each predictor is trained on an i.i.d. dataset of b samples drawn from query history, where b is the bootstrap size.
    If the training set selected from history does not contain all labels, no predictors are trained and an empty list
    is returned.

    Note: The bootstrapping process occurs as follows. To generate a diverse hypothesis space, each predictor is
    trained on a set of examples selected i.i.d. (at random with replacement) from the same training set.

    :param history: Dictionary containing query history. Expected to be empty or contain four keys: 'X','y','c','Q'.
                    'X' contains a numpy array (n,2) where each row represents a sample and n is the number of
                    queries made so far. 'y' contains a numpy array (n,) where each row represents a sample label.
                    'c' contains a list where each element represents the probability a sample was selected for
                    labeling. 'Q' contains a list where each element represents the 0-1 result of a coin flip.
    :param bootstrap_size: Number of samples to be used in bootstrapping.
    :param num_predictors: Number of predictors to be trained.
    :param labels: List of all possible labels for the data set.
    :return: List of predictors trained using the training set.
    """

    # limit training set to predetermined portion of history
    X_train, y_train = _bootstrap_select_bootstrap_training_set(history, bootstrap_size)

    predictors = []
    if _bootstrap_y_has_all_labels(y_train, labels):

        # train predictors
        for i in range(num_predictors):
            X_iid, y_iid = _bootstrap_select_iid_training_set(X_train, y_train, labels)
            lr = LogisticRegression().fit(X_iid, y_iid)
            predictors.append(lr)

    return predictors


def bootstrap(x_t: np.ndarray, history: dict, loss_function: Any, bootstrap_size: int = 10, num_predictors: int = 10,
              labels: list = [0, 1], p_min: float = 0.1) -> float:
    """
    This function implements Algorithm 3 from the paper by Beygelzimer et al. See https://arxiv.org/pdf/0812.4952.pdf.
    Uses bootstrapping to generate a hypothesis space and calculates rejection threshold probability p_t for unlabeled
    sample x_t:

    p_t = p_min + (1 - p_min) * max_{y in Y, h_i, h_j in H}{l(h_i(x),y) - l(h_j(x),y)}

    :param x_t: (2,) numpy array representing sample data point.
    :param history: Dictionary containing query history. Expected to be empty or contain four keys: 'X','y','c','Q'.
                    'X' contains a numpy array (n,2) where each row represents a sample and n is the number of
                    queries made so far. 'y' contains a numpy array (n,) where each row represents a sample label.
                    'c' contains a list where each element represents the probability a sample was selected for
                    labeling. 'Q' contains a list where each element represents the 0-1 result of a coin flip.
    :param bootstrap_size: Number of samples to be used in bootstrapping.
    :param num_predictors: Number of predictors to be trained.
    :param labels: List of all possible labels for the data set.
    :param p_min: Minimum probability for selecting a sample for labeling. Default value is 0.1.
    :param loss_function: Python function which calculates loss normalized to [0,1] range. The following signature is
                          expected: function_name(predictor:Any, x_t:np.ndarray, label:Any, labels:list) -> Float.
                          predictor: sklearn model;
                          x_t: (2,) numpy array representing sample data point;
                          label: single label from list of labels;
                          labels: List of all possible labels for the data set.
    :return: Float representing the rejection threshold probability. Value is in [0,1].
    """
    # actions depend on current training size and bootstrap size
    if 'X' in history:
        training_size = len(history['X'])
    else:
        training_size = 0

    if training_size < bootstrap_size:  # bootstrapping process is not complete

        p_t = 1.0  # sample will be used for bootstrapping and should be added to selected set

    else:
        # create bootstrapped committee of predictors
        predictors = _bootstrap_train_predictors(history, bootstrap_size, num_predictors, labels)

        # calculate probability
        p_t = _bootstrap_calculate_p_t(x_t, predictors, labels, p_min, loss_function)

    return p_t
