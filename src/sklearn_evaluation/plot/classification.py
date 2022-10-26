"""
Plotting functions for classifier models
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from ..telemetry import SKLearnEvaluationLogger, telemetry
from ..plot.matplotlib import bar
from ..metrics import precision_at
from .. import compute
from ..util import is_column_vector, is_row_vector, default_heatmap
from ..report.serialize import figure2html


class Plot:

    def _repr_html_(self):
        return figure2html(self.figure)


def _confusion_matrix_add(first, second, ax, target_names):
    # Adapted from: https://stackoverflow.com/a/63531813/709975

    # TODO: validate first and second have the same shape
    M, N = first.shape
    x = np.arange(M + 1)
    y = np.arange(N + 1)

    xs, ys = np.meshgrid(x, y)

    zs = (xs * ys) % 10
    zs = zs[:-1, :-1].ravel()

    max_ = np.max([first.max(), second.max()])

    triangles1 = [(i + j * (M + 1), i + 1 + j * (M + 1), i + (j + 1) * (M + 1))
                  for j in range(N) for i in range(M)]
    triangles2 = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1),
                   i + (j + 1) * (M + 1)) for j in range(N) for i in range(M)]
    triang1 = Triangulation(xs.ravel() - 0.5, ys.ravel() - 0.5, triangles1)
    triang2 = Triangulation(xs.ravel() - 0.5, ys.ravel() - 0.5, triangles2)

    cmap = default_heatmap()

    img1 = ax.tripcolor(triang1, first.ravel(), cmap=cmap, vmax=max_)
    _ = ax.tripcolor(triang2, second.ravel(), cmap=cmap, vmax=max_)
    # ax.figure.colorbar(img1)

    ax.set_xlim(x[0] - 0.5, x[-1] - 0.5)
    ax.set_ylim(y[-1] - 0.5, y[0] - 0.5)

    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(target_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(target_names)

    for pad, arr in ((-1 / 5, first), (1 / 5, second)):
        for (y, x), v in np.ndenumerate(arr):
            try:
                label = '{:.2}'.format(v)
            except Exception:
                label = v

            ax.text(x + pad,
                    y + pad,
                    label,
                    horizontalalignment='center',
                    verticalalignment='center')

    title = 'Confusion matrix (compare)'
    ax.set_title(title)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')


class ConfusionMatrixSub(Plot):

    def __init__(self, cm, target_names) -> None:
        self.figure = Figure()
        ax = self.figure.add_subplot()
        _plot_cm(cm,
                 cmap=default_heatmap(),
                 ax=ax,
                 target_names=target_names,
                 normalize=False)


class ConfusionMatrixAdd(Plot):

    def __init__(self, a, b, target_names) -> None:
        self.figure = Figure()
        ax = self.figure.add_subplot()
        _confusion_matrix_add(a, b, ax=ax, target_names=target_names)


class ConfusionMatrix(Plot):

    @telemetry.log_call('sklearn-evaluation-ConfusionMatrix')
    def __init__(self, y_true, y_pred, target_names=None, normalize=False):
        self.cm = _confusion_matrix(y_true, y_pred, normalize)
        self.figure = Figure()
        ax = self.figure.add_subplot()

        self.target_names, cmap, ax = _confusion_matrix_validate(y_true,
                                                                 y_pred,
                                                                 target_names,
                                                                 cmap=None,
                                                                 ax=ax)
        cm = _confusion_matrix(y_true, y_pred, normalize)
        _plot_cm(cm, cmap, ax, self.target_names, normalize)

    def __sub__(self, other):
        cm = self.cm - other.cm
        return ConfusionMatrixSub(cm, self.target_names)

    def __add__(self, other):
        return ConfusionMatrixAdd(self.cm, other.cm, self.target_names)


def _confusion_matrix(y_true, y_pred, normalize):
    cm = sk_confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm


@SKLearnEvaluationLogger.log(feature='plot')
def confusion_matrix(
        y_true,
        y_pred,
        target_names=None,
        normalize=False,
        cmap=None,
        ax=None):
    """
    Plot confusion matrix.

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

    Notes
    -----
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html


    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/confusion_matrix.py

    """
    target_names, cmap, ax = _confusion_matrix_validate(
        y_true, y_pred, target_names, cmap, ax)
    cm = _confusion_matrix(y_true, y_pred, normalize)
    return _plot_cm(cm, cmap, ax, target_names, normalize)


def _confusion_matrix_validate(y_true, y_pred, target_names, cmap, ax):
    if any((val is None for val in (y_true, y_pred))):
        raise ValueError("y_true and y_pred are needed to plot confusion "
                         "matrix")

    # calculate how many names you expect
    values = set(y_true).union(set(y_pred))
    expected_len = len(values)

    if target_names and (expected_len != len(target_names)):
        raise ValueError(
            ('Data cointains {} different values, but target'
             ' names contains {} values.'.format(expected_len,
                                                 len(target_names))))

    # if the user didn't pass target_names, create generic ones
    if not target_names:
        values = list(values)
        values.sort()
        target_names = ['Class {}'.format(v) for v in values]

    np.set_printoptions(precision=2)

    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = default_heatmap()

    return target_names, cmap, ax


def _plot_cm(cm, cmap, ax, target_names, normalize):
    # this (y, x) may sound counterintuitive. The reason is that
    # in a matrix cell (i, j) is in row=i and col=j, translating that
    # to an x, y plane (which matplotlib uses to plot), we need to use
    # i as the y coordinate (how many steps down) and j as the x coordinate
    # how many steps to the right.
    for (y, x), v in np.ndenumerate(cm):
        try:
            label = '{:.2}'.format(v)
        except Exception:
            label = v
        ax.text(x,
                y,
                label,
                horizontalalignment='center',
                verticalalignment='center')

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im)
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
@SKLearnEvaluationLogger.log(feature='plot')
def feature_importances(data,
                        top_n=None,
                        feature_names=None,
                        orientation='horizontal',
                        ax=None):
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
    orientation: ('horizontal', 'vertical')
        Bar plot orientation
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
    if data is None:
        raise ValueError('data is needed to plot feature importances. '
                         'When plotting using the evaluator you need to pass '
                         'an estimator ')

    # If no feature_names is provided, assign numbers
    res = compute.feature_importances(data, top_n, feature_names)

    ax = bar.plot(res.importance,
                  orientation,
                  res.feature_name,
                  error=None if not hasattr(res, 'std_') else res.std_)
    ax.set_title("Feature importances")
    return ax


@SKLearnEvaluationLogger.log(feature='plot')
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
    if any((val is None for val in (y_true, y_score))):
        raise ValueError('y_true and y_score are needed to plot precision at '
                         'proportions')

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
