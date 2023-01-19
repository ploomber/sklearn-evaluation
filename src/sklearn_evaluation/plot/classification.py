"""
Plotting functions for classifier models
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from sklearn_evaluation import __version__
from sklearn_evaluation.telemetry import SKLearnEvaluationLogger
from sklearn_evaluation.plot.matplotlib import bar
from sklearn_evaluation.metrics import precision_at
from sklearn_evaluation import compute
from sklearn_evaluation.util import is_column_vector, is_row_vector, default_heatmap
from sklearn_evaluation.plot.plot import AbstractPlot, AbstractComposedPlot
from sklearn_evaluation.plot import _matrix
from ploomber_core.exceptions import modify_exceptions


class ConfusionMatrixSub(AbstractComposedPlot):
    """A composed plot to compare the difference between two confusion matrices"""

    def __init__(self, cm, target_names) -> None:
        self.cm = cm
        self.target_names = target_names

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        _plot_cm(
            self.cm,
            cmap=default_heatmap(),
            ax=ax,
            target_names=self.target_names,
            normalize=False,
        )

        return ax


class ConfusionMatrixAdd(AbstractComposedPlot):
    """A composed plot to compare two confusion matrices"""

    def __init__(self, a, b, target_names) -> None:
        self.a = a
        self.b = b
        self.target_names = target_names

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        _matrix.add(self.a, self.b, ax, invert_axis=True)

        tick_marks = np.arange(len(self.target_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(self.target_names)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(self.target_names)

        title = "Confusion matrix (compare)"
        ax.set_title(title)

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

        return ax


class ConfusionMatrix(AbstractPlot):
    """
    Plot confusion matrix.

    .. seealso:: :func:`confusion_matrix`

    Examples
    --------
    .. plot:: ../examples/confusion_matrix_oop.py

    Notes
    -----
    .. versionchanged:: 0.9
        Added ``cmap`` argument
    """

    @SKLearnEvaluationLogger.log(feature="plot", action="confusion-matrix-init")
    @modify_exceptions
    def __init__(self, cm, *, target_names=None, normalize=False, cmap=None):
        self.cm = cm
        self.target_names = target_names
        self.normalize = normalize
        self.cmap = _confusion_matrix_init_defaults(cmap=cmap)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        _plot_cm(self.cm, self.cmap, ax, self.target_names, self.normalize)

        self.ax_ = ax
        self.figure_ = ax.figure

        return self

    @SKLearnEvaluationLogger.log(feature="plot", action="confusion-matrix-sub")
    def __sub__(self, other):
        cm = self.cm - other.cm
        obj = ConfusionMatrixSub(cm, self.target_names)
        obj.plot()
        return obj

    @SKLearnEvaluationLogger.log(feature="plot", action="confusion-matrix-add")
    def __add__(self, other):
        obj = ConfusionMatrixAdd(self.cm, other.cm, self.target_names)
        obj.plot()
        return obj

    def _get_data(self):
        return {
            "class": "sklearn_evaluation.plot.ConfusionMatrix",
            "cm": self.cm.tolist(),
            "normalize": self.normalize,
            "target_names": self.target_names,
            "version": __version__,
        }

    @classmethod
    def from_dump(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cm = np.array(data["cm"])
        normalize = data["normalize"]
        target_names = data["target_names"]
        return cls(cm=cm, target_names=target_names, normalize=normalize).plot()

    @classmethod
    @modify_exceptions
    def from_raw_data(
        cls, y_true, y_pred, target_names=None, normalize=False, cmap=None
    ):
        """

        Notes
        -----
        .. versionchanged:: 0.9
            Added ``cmap`` argument.
        """
        target_names, cmap = _confusion_matrix_validate(
            y_true, y_pred, target_names, cmap=cmap
        )
        cm = _confusion_matrix(y_true, y_pred, normalize)
        return cls(cm, target_names=target_names, normalize=normalize, cmap=cmap).plot()

    @classmethod
    def _from_data(cls, target_names, normalize, cm):
        return cls(
            cm=np.array(cm),
            target_names=target_names,
            normalize=normalize,
        )


def _confusion_matrix(y_true, y_pred, normalize):
    cm = sk_confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    return cm


@SKLearnEvaluationLogger.log(feature="plot")
@modify_exceptions
def confusion_matrix(
    y_true, y_pred, target_names=None, normalize=False, cmap=None, ax=None
):
    """
    Plot confusion matrix.

    .. seealso:: :class:`ConfusionMatrix`

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
    .. plot:: ../examples/confusion_matrix.py

    """
    return ConfusionMatrix.from_raw_data(
        y_true=y_true,
        y_pred=y_pred,
        target_names=target_names,
        normalize=normalize,
        cmap=cmap,
    ).ax_


def _confusion_matrix_validate_predictions(y_true, y_pred, target_names):
    """Validate input prediction data for a confusion matrix"""
    if any((val is None for val in (y_true, y_pred))):
        raise ValueError("y_true and y_pred are needed to plot confusion " "matrix")

    # calculate how many names you expect
    values = set(y_true).union(set(y_pred))
    expected_len = len(values)

    if target_names and (expected_len != len(target_names)):
        raise ValueError(
            (
                "Data cointains {} different values, but target"
                " names contains {} values.".format(expected_len, len(target_names))
            )
        )

    if not target_names:
        values = list(values)
        values.sort()
        target_names = ["Class {}".format(v) for v in values]

    return target_names


def _confusion_matrix_init_defaults(cmap):
    """Initialize default values for confusion matrix"""
    # if the user didn't pass target_names, create generic ones
    np.set_printoptions(precision=2)

    if cmap is None:
        cmap = default_heatmap()

    return cmap


def _confusion_matrix_validate(y_true, y_pred, target_names, cmap):
    """Validate values for confusion matrix and initialize defaults"""
    target_names = _confusion_matrix_validate_predictions(y_true, y_pred, target_names)
    cmap = _confusion_matrix_init_defaults(cmap)
    return target_names, cmap


def _add_values_to_matrix(m, ax):
    # this (y, x) may sound counterintuitive. The reason is that
    # in a matrix cell (i, j) is in row=i and col=j, translating that
    # to an x, y plane (which matplotlib uses to plot), we need to use
    # i as the y coordinate (how many steps down) and j as the x coordinate
    # how many steps to the right.
    for (y, x), v in np.ndenumerate(m):
        try:
            label = "{:.2}".format(v)
        except Exception:
            label = v
        ax.text(x, y, label, horizontalalignment="center", verticalalignment="center")


def _plot_cm(cm, cmap, ax, target_names, normalize):
    """Adds confusion matrix graphical elements to a ``matplotlib.Axes`` object"""
    _add_values_to_matrix(cm, ax)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im)
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(target_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(target_names)

    title = "Confusion matrix"

    if normalize:
        title += " (normalized)"

    ax.set_title(title)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    return ax


# Receiver operating characteristic (ROC) with cross validation
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#example-model-selection-plot-roc-crossval-py


# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
@SKLearnEvaluationLogger.log(feature="plot")
@modify_exceptions
def feature_importances(
    data, top_n=None, feature_names=None, orientation="horizontal", ax=None
):
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
    .. plot:: ../examples/feature_importances.py

    """
    if data is None:
        raise ValueError(
            "data is needed to plot feature importances. "
            "When plotting using the evaluator you need to pass "
            "an estimator "
        )

    # If no feature_names is provided, assign numbers
    res = compute.feature_importances(data, top_n, feature_names)

    ax = bar.plot(
        res.importance,
        orientation,
        res.feature_name,
        error=None if not hasattr(res, "std_") else res.std_,
    )
    ax.set_title("Feature importances")
    return ax


@SKLearnEvaluationLogger.log(feature="plot")
@modify_exceptions
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
        raise ValueError(
            "y_true and y_score are needed to plot precision at " "proportions"
        )

    if ax is None:
        _, ax = plt.subplots()

    y_score_is_vector = is_column_vector(y_score) or is_row_vector(y_score)
    if not y_score_is_vector:
        y_score = y_score[:, 1]

    # Calculate points
    proportions = [0.01 * i for i in range(1, 101)]
    precs_and_cutoffs = [precision_at(y_true, y_score, p) for p in proportions]
    precs, cutoffs = zip(*precs_and_cutoffs)

    # Plot and set nice defaults for title and axis labels
    ax.plot(proportions, precs)
    ax.set_title("Precision at various proportions")
    ax.set_ylabel("Precision")
    ax.set_xlabel("Proportion")
    ticks = [0.1 * i for i in range(1, 11)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.set_ylim([0, 1.0])
    ax.set_xlim([0, 1.0])
    return ax
