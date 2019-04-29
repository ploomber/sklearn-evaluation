import numpy as np
import matplotlib.pyplot as plt
from sklearn_evaluation.metrics import ThresholdBinarizer


def metrics_at_thresholds(fn, y_true, y_score, n_thresholds=10, start=0.0,
                          ax=None):
    """Plot metrics at increasing thresholds

    Examples
    --------
    >>> from sklearn_evaluation.metrics import ThresholdBinarizer
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.metrics import precision_score, recall_score, f1_score
    >>> import numpy as np
    >>> y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    >>> y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    >>> metrics_at_thresholds([precision_score, recall_score,
    >>>                        f1_score], y_true, y_score)
    """
    if ax is None:
        ax = plt.gca()

    th, m = ThresholdBinarizer.metric_at(fn, y_true, y_score, n_thresholds,
                                         start)

    ax.plot(th, np.array(m).T, '.--')
    ax.legend([fn_.__name__ for fn_ in fn])
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric value')
    ax.grid()

    return ax
