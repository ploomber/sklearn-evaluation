from jinja2 import Environment, PackageLoader
import time
import numpy as np
from sklearn_evaluation import plot
from sklearn.metrics import auc


def jinja_env():
    env = Environment(
        loader=PackageLoader("sklearn_evaluation", "assets/report"),
    )
    return env


class Range(object):
    """
    Set float range
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def in_range(self, n) -> bool:
        """
        Checks if n in range
        """
        return self.min <= n and self.max >= n


# Model evaluator utils
def check_array_balance(array) -> bool:
    """
    Check if array is balanced
    """
    classes, counts = np.unique(array, return_counts=True)
    n_classes = len(classes)
    n_values = len(array)
    expected_balance = 1 / n_classes

    weights = list(map(lambda count: count / n_values, counts))
    balance_threshold = 0.05
    expected_range = Range(
        expected_balance - balance_threshold, expected_balance + balance_threshold
    )

    return all(expected_range.in_range(w) for w in weights)


# Model comparer utils


def get_roc_auc(y_test, y_score) -> list:
    """
    Returns list of roc auc
    """
    roc_auc = []
    roc = plot.ROC.from_raw_data(y_test, y_score)
    for i in range(len(roc.fpr)):
        roc_auc.append(auc(roc.fpr[i], roc.tpr[i]))

    return roc_auc


def get_model_prediction_time(model, X_test) -> float:
    """
    Returns model predict time in seconds
    """
    start = time.time()
    model.predict(X_test)
    end = time.time()
    eval_time = end - start  # in seconds
    return eval_time


def validate_args_not_none(func):
    """
    Decorator to validate no None args passed
    """

    def wrapper(*args, **kw):
        if not any(elem is None for elem in args):
            return func(*args, **kw)

    return wrapper
