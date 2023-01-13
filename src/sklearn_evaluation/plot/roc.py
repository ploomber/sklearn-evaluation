import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from ..telemetry import SKLearnEvaluationLogger
from ..util import is_column_vector, is_row_vector
from sklearn_evaluation import __version__
import json
from pathlib import Path
from sklearn_evaluation.plot.plot import AbstractPlot, AbstractComposedPlot
from ploomber_core.exceptions import modify_exceptions


@modify_exceptions
def roc(y_true, y_score, ax=None):
    """Plot ROC curve

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).

    y_score : array-like, shape = [n_samples] or [n_samples, 2] for binary
        classification or [n_samples, n_classes] for multiclass
        Target scores (estimator predictions).

    ax: matplotlib Axes, default: None
        Axes object to draw the plot onto, otherwise uses current Axes

    Notes
    -----
    It is assumed that the y_score parameter columns are in order.
    For example, if ``y_true = [2, 2, 1, 0, 0, 1, 2]``, then the
    first column in y_score must contain the scores for class 0,
    second column for class 1 and so on.

    Examples
    --------
    .. plot:: ../examples/roc.py
    """
    # support old API
    r = ROC.from_raw_data(y_true, y_score, ax=ax)
    return r.ax


def _set_ax_settings(ax):
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC")
    ax.legend(loc="best")


def _roc_curve_multi(y_true, y_score):
    """Compute micro-average ROC curve"""
    return roc_curve(y_true.ravel(), y_score.ravel())


def _plot_roc(fpr, tpr, ax, label=None):
    """
    Plot ROC curve

    Parameters
    ----------
    fpr : ndarray of shape (>2,)
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= `thresholds[i]`

    tpr : ndarray of shape (>2,)
        Increasing true positive rates such that element `i` is the true
        positive rate of predictions with score >= `thresholds[i]`

    ax: matplotlib Axes
        Axes object to draw the plot onto

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Notes
    -----
    .. versionadded:: 0.8.4
    """
    roc_auc = auc(fpr, tpr)

    label = label or "ROC curve"

    ax.plot(fpr, tpr, label=(f"{label} (area = {roc_auc:0.2f})"))

    _set_ax_settings(ax)
    return ax


def _generate_plot_from_fpr_tpr_lists(fpr, tpr, ax, label=None):
    """
    Draws a plot for every list of values i.e tpr[i] and fpr[i].
    """
    for i in range(len(fpr)):
        fpr_ = fpr[i]
        tpr_ = tpr[i]
        label_ = label[i] if label is not None and len(label) > 0 else None
        _plot_roc(fpr_, tpr_, ax, label=label_)


class ROCAdd(AbstractComposedPlot):
    """Generate a new plot with overlapping ROC curves (roc1 + roc2)

    Parameters
    ----------
    a : ROC
        ROC plot

    b : ROC
        ROC plot

    Examples
    --------
    .. plot:: ../../examples/roc_add.py

    Notes
    -----
    .. versionadded:: 0.8.4

    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def plot(self, ax=None):
        a = self.a
        b = self.b

        if ax is None:
            _, ax = plt.subplots()

        _generate_plot_from_fpr_tpr_lists(a.fpr, a.tpr, ax, label=a.label)

        # mark the added curve with 2
        if b.label is not None and len(b.label) > 0:
            b_label = [label + " 2" for label in b.label]
        else:
            b_label = ["ROC curve 2"]

        _generate_plot_from_fpr_tpr_lists(b.fpr, b.tpr, ax, label=b_label)

        self.ax_ = ax
        self.figure_ = ax.figure

        return self


class ROC(AbstractPlot):
    """Plot ROC curve

    Parameters
    ----------
    fpr : ndarray of shape (>2,), list of lists or list of numbers
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= `thresholds[i]`.

    tpr : ndarray of shape (>2,), list of lists or list of numbers
        Increasing true positive rates such that element `i` is the true
        positive rate of predictions with score >= `thresholds[i]`.

    label : list of str, default: None
        Set curve labels

    ax: matplotlib Axes, default: None
        Axes object to draw the plot onto, otherwise uses current Axes

    Examples
    --------
    .. plot:: ../examples/roc_binary_classification.py

    .. plot:: ../examples/roc_comparison.py

    .. plot:: ../examples/roc_multi_classification.py

    Notes
    -----
    .. versionadded:: 0.8.4
    """

    @SKLearnEvaluationLogger.log(feature="plot", action="roc-init")
    @modify_exceptions
    def __init__(self, fpr, tpr, label=None, ax=None):
        if fpr is None or tpr is None:
            raise TypeError("fpr and tpr must be defined.")

        if type(fpr) != type(tpr):
            raise TypeError(
                "fpr and tpr must be the same type. "
                f"Recevied: fpr {type(fpr)} != tpr {type(tpr)}"
            )

        if len(fpr) == 0 or len(tpr) == 0:
            raise ValueError("fpr and tpr must not be empty")

        if not isinstance(fpr[0], (list, np.ndarray)):
            fpr = [fpr]
            tpr = [tpr]

        if len(fpr) != len(tpr):
            raise ValueError(
                "fpr and tpr lengths should correspond. "
                f"Recevied: fpr {len(fpr)} != tpr {len(tpr)}"
            )

        for i in range(len(fpr)):
            fpr_ = fpr[i]
            tpr_ = tpr[i]

            if len(fpr_) != len(tpr_):
                raise ValueError(
                    "fpr and tpr lengths should correspond. "
                    f"Recevied: fpr {len(fpr_)} != tpr {len(tpr_)} at index {i}"
                )

        self.fpr = fpr
        self.tpr = tpr
        self.label = label
        self.ax = ax

    def __sub__(self):
        raise NotImplementedError("Not applicable for ROC")

    @SKLearnEvaluationLogger.log(feature="plot", action="roc-add")
    def __add__(self, other):
        roc_add_result = ROCAdd(self, other)
        roc_add_result.plot()
        return roc_add_result

    def _get_data(self):
        fpr = self.fpr
        tpr = self.tpr

        # for serialization
        for i in range(len(fpr)):
            if isinstance(fpr[i], np.ndarray):
                fpr[i] = fpr[i].tolist()
                tpr[i] = tpr[i].tolist()

        return {
            "class": "sklearn_evaluation.plot.ROC",
            "version": __version__,
            "fpr": fpr,
            "tpr": tpr,
        }

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        _generate_plot_from_fpr_tpr_lists(self.fpr, self.tpr, ax, label=self.label)

        self.ax = ax
        self.figure_ = ax.figure

        return self

    @classmethod
    def from_dump(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        fpr = data["fpr"]
        tpr = data["tpr"]

        return cls(fpr, tpr).plot()

    @classmethod
    @modify_exceptions
    def from_raw_data(cls, y_true, y_score, ax=None):
        fpr, tpr, label = cls._calculate_plotting_data(y_true, y_score)
        return cls(fpr, tpr, label=label, ax=ax).plot()

    @staticmethod
    @modify_exceptions
    def _calculate_plotting_data(y_true, y_score):
        """
        Plot ROC curve
        Parameters
        ----------
        y_true : array-like, shape = [n_samples]
            Correct target values (ground truth).

        y_score : array-like, shape = [n_samples] or [n_samples, 2] for binary
            classification or [n_samples, n_classes] for multiclass
            Target scores (estimator predictions).

        Returns
        -------
        fpr : list of lists with fpr values

        tpr : list of lists with tpr values

        label : list of str for curves label
        """
        if any((val is None for val in (y_true, y_score))):
            raise ValueError("y_true and y_score are needed to plot ROC")

        label = []

        # get the number of classes based on the shape of y_score
        y_score_is_vector = is_column_vector(y_score) or is_row_vector(y_score)
        if y_score_is_vector:
            n_classes = 2
        else:
            _, n_classes = y_score.shape

        if n_classes > 2:
            # convert y_true to binary format
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

            fpr = []
            tpr = []

            avg_fpr, avg_tpr, _ = _roc_curve_multi(y_true_bin, y_score)

            label.append("micro-average ROC curve")
            fpr.append(avg_fpr)
            tpr.append(avg_tpr)

            for i in range(n_classes):
                fpr_, tpr_, _ = roc_curve(y_true_bin[:, i], y_score[:, i])

                label.append("ROC curve")
                fpr.append(fpr_)
                tpr.append(tpr_)
        else:
            if y_score_is_vector:
                fpr, tpr, _ = roc_curve(y_true, y_score)
            else:
                fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])

            fpr = [fpr]
            tpr = [tpr]

        return fpr, tpr, label
