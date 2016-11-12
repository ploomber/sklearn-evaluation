"""Plotting functions."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from ..metrics import precision_at
from .. import compute
from ..util import is_column_vector, is_row_vector, default_heatmap


# Confusion matrix
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def confusion_matrix(y_true, y_pred, target_names=None, normalize=False,
                     cmap=None, ax=None):
    """
    Plot confustion matrix.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).
    y_pred : array-like, shape = [n_samples]
        Target predicted classes (estimator predictions).
    target_names : list
        List containing the names of the target classes. List must be in order
        e.g. ``['Label for class 0', 'Label for class 1']``. If ``None``,
        generic labels will be generated e.g. ``['Class 0', 'Class 1']``
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes
    normalize : bool
        Normalize the confusion matrix
    cmap : matplotlib Colormap
        If ``None`` uses a modified version of matplotlib's OrRd colormap.


    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/confusion_matrix.py

    """
    # calculate how many names you expect
    values = set(y_true).union(set(y_pred))
    expected_len = len(values)

    if target_names and (expected_len != len(target_names)):
        raise ValueError(('Data cointains {} different values, but target'
                         ' names contains {} values.'.format(expected_len,
                                                             len(target_names)
                                                             )))

    # if the user didn't pass target_names, create generic ones
    if not target_names:
        values = list(values)
        values.sort()
        target_names = ['Class {}'.format(v) for v in values]

    cm = sk_confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)

    if ax is None:
        ax = plt.gca()

    # this (y, x) may sound counterintuitive. The reason is that
    # in a matrix cell (i, j) is in row=i and col=j, translating that
    # to an x, y plane (which matplotlib uses to plot), we need to use
    # i as the y coordinate (how many steps down) and j as the x coordinate
    # how many steps to the right.
    for (y, x), v in np.ndenumerate(cm):
        try:
            label = '{:.2}'.format(v)
        except:
            label = v
        ax.text(x, y, label, horizontalalignment='center',
                verticalalignment='center')

    if cmap is None:
        cmap = default_heatmap()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(target_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(target_names)

    title = 'Confusion matrix'
    if normalize:
        title += ' (normalized)'
    ax.set_title(title)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return ax


# Receiver operating characteristic (ROC) with cross validation
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#example-model-selection-plot-roc-crossval-py


# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
def feature_importances(data, top_n=None, feature_names=None, ax=None):
    """
    Get and order feature importances from a scikit-learn model
    or from an array-like structure. If data is a scikit-learn model with
    sub-estimators (e.g. RandomForest, AdaBoost) the function will compute the
    standard deviation of each feature.

    Parameters
    ----------
    data : sklearn model or array-like structure
        Object to get the data from.
    top_n : int
        Only get results for the top_n features.
    feature_names : array-like
        Feature names
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/feature_importances.py

    """
    # If no feature_names is provided, assign numbers
    res = compute.feature_importances(data, top_n, feature_names)
    # number of features returned
    n_feats = len(res)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Feature importances")

    try:
        ax.bar(range(n_feats), res.importance, yerr=res.std_, color='red',
               align="center")
    except:
        ax.bar(range(n_feats), res.importance, color='red',
               align="center")

    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(res.feature_name)
    ax.set_xlim([-1, n_feats])
    return ax


def precision_at_proportions(y_true, y_score, ax=None):
    """
    Plot precision values at different proportions.

    Parameters
    ----------
    y_true : array-like
        Correct target values (ground truth).
    y_score : array-like
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

    y_score_is_vector = is_column_vector(y_score) or is_row_vector(y_score)
    if not y_score_is_vector:
        y_score = y_score[:, 1]

    # Calculate points
    proportions = [0.01 * i for i in range(1, 101)]
    precs_and_cutoffs = [precision_at(y_true, y_score, p) for p in proportions]
    precs, cutoffs = zip(*precs_and_cutoffs)

    # Plot and set nice defaults for title and axis labels
    ax.plot(proportions, precs)
    ax.set_title('Precision at various proportions')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Proportion')
    ticks = [0.1 * i for i in range(1, 11)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.set_ylim([0, 1.0])
    ax.set_xlim([0, 1.0])
    return ax
