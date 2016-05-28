import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


def precision_recall(y_true, y_score, ax=None):
    """
    Plot precision-recall curve.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_samples, n_classes]
        Target scores (estimator predictions).
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    if ax is None:
        ax = plt.gca()

    # get the number of classes from y_score
    _, n_classes = y_score.shape

    # check data shape?

    if n_classes > 2:
        # convert y_true to binary format
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        _precision_recall_multi(y_true_bin, y_score, ax=ax)
        for i in range(n_classes):
            _precision_recall(y_true_bin[:, i], y_score[:, i], ax=ax)
    else:
        _precision_recall(y_true, y_score, ax)

    # raise error if n_classes = 1?
    return ax


def _precision_recall(y_true, y_score, ax=None):
    """
    Plot precision-recall curve.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_samples]
        Target scores (estimator predictions).
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)

    if ax is None:
        ax = plt.gca()

    ax.plot(recall, precision, label='Precision-Recall curve')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(('Precision-Recall curve: AUC={0:0.2f}'
                  .format(average_precision)))
    ax.legend(loc="best")
    return ax


def _precision_recall_multi(y_true, y_score, ax=None):
    """
    Plot precision-recall curve.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples, n_classes]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_samples, n_classes]
        Target scores (estimator predictions).
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    # Compute micro-average ROC curve and ROC area
    precision, recall, _ = precision_recall_curve(y_true.ravel(),
                                                  y_score.ravel())

    avg_prec = average_precision_score(y_true, y_score, average="micro")

    if ax is None:
        ax = plt.gca()

    ax.plot(recall, precision,
            label=('micro-average Precision-recall curve (area = {0:0.2f})'
                   .format(avg_prec)))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall')
    ax.legend(loc="lower right")
    return ax
