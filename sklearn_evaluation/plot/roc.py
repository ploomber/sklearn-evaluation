import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from ..util import is_column_vector, is_row_vector


def roc(y_true, y_score, ax=None):
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_samples] or [n_samples, 2] for binary
              classification or [n_samples, n_classes] for multiclass

        Target scores (estimator predictions).
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Notes
    -----
    It is assumed that the y_score parameter columns are in order. For example,
    if ``y_true = [2, 2, 1, 0, 0, 1, 2]``, then the first column in y_score
    must countain the scores for class 0, second column for class 1 and so on.


    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/roc.py

    """
    if ax is None:
        ax = plt.gca()

    # get the number of classes based on the shape of y_score
    y_score_is_vector = is_column_vector(y_score) or is_row_vector(y_score)
    if y_score_is_vector:
        n_classes = 2
    else:
        _, n_classes = y_score.shape

    # check data shape?

    if n_classes > 2:
        # convert y_true to binary format
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        _roc_multi(y_true_bin, y_score, ax=ax)
        for i in range(n_classes):
            _roc(y_true_bin[:, i], y_score[:, i], ax=ax)
    else:
        if y_score_is_vector:
            _roc(y_true, y_score, ax)
        else:
            _roc(y_true, y_score[:, 1], ax)

    # raise error if n_classes = 1?
    return ax


def _roc(y_true, y_score, ax=None):
    """
    Plot ROC curve for binary classification.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_samples]
        Target scores (estimator predictions).
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    # check dimensions

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, label=('ROC curve (area = {0:0.2f})'.format(roc_auc)))
    _set_ax_settings(ax)
    return ax


def _roc_multi(y_true, y_score, ax=None):
    """
    Plot ROC curve for multi classification.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples, n_classes]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_samples, n_classes]
        Target scores (estimator predictions).
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    if ax is None:
        ax = plt.gca()

    ax.plot(fpr, tpr, label=('micro-average ROC curve (area = {0:0.2f})'
                             .format(roc_auc)))
    _set_ax_settings(ax)
    return ax


def _set_ax_settings(ax):
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC')
    ax.legend(loc="best")
