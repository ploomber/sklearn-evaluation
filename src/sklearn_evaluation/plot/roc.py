import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from ..telemetry import SKLearnEvaluationLogger
from ..util import is_column_vector, is_row_vector
from ..plot.plot import Plot
from sklearn_evaluation import __version__
import json
from pathlib import Path
from warnings import warn  # noqa


def roc(y_true, y_score, ax=None):
    # Support old api
    """
    Plot ROC curve
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
    r = ROC(y_true, y_score, ax=ax)
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
    # Compute micro-average ROC curve
    return roc_curve(y_true.ravel(), y_score.ravel())


def _plot_roc(fpr, tpr, ax, curve_label="ROC curve", line_color=None):
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

    ax.plot(
        fpr, tpr, label=(f"{curve_label} (area = {roc_auc:0.2f})"), color=line_color
    )

    _set_ax_settings(ax)
    return ax


def _plot_roc_multi_classification(
    avg_fpr, avg_tpr, roc_rates_n_classes, ax, curve_label="ROC curve"
):
    """
    Plot ROC curve for multi classification

    Parameters
    ----------
    avg_fpr : ndarray of shape (>2,)
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= `thresholds[i]`

    avg_tpr : ndarray of shape (>2,)
        Increasing true positive rates such that element `i` is the true
        positive rate of predictions with score >= `thresholds[i]`

    roc_rates_n_classes : list of dictionaries with 'fpr' and 'tpr'
        i.e : [{'fpr' : [0.0, 0.2, 0.4, 0.4, 0.6, 1.0],
        'tpr' : [0.0, 0.2, 0.4, 1.0, 1.0, 1.0]}]

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
    _plot_roc(avg_fpr, avg_tpr, ax, curve_label=f"micro-average {curve_label}")
    for d in roc_rates_n_classes:
        _plot_roc(np.array(d["fpr"]), np.array(d["tpr"]), ax, curve_label=curve_label)
    return ax


class ROCAdd(Plot):
    """
    Generate a new plot with overlapping ROC curves (roc1 + roc2)

    Parameters
    ----------
    a : ROC plot

    b : ROC plot

    Examples
    --------
    .. plot:: ../../examples/roc_add.py

    Notes
    -----
    .. versionadded:: 0.8.4
    """

    def __init__(self, a, b):
        self.figure = plt.figure()
        ax = self.figure.add_subplot()
        added_curve_label = "ROC curve 2"

        if hasattr(a, "roc_rates_n_classes"):
            self.ax = _plot_roc_multi_classification(
                a.fpr, a.tpr, a.roc_rates_n_classes, ax
            )
        else:
            _plot_roc(a.fpr, a.tpr, ax=ax)

        if hasattr(b, "roc_rates_n_classes"):
            self.ax = _plot_roc_multi_classification(
                b.fpr, b.tpr, b.roc_rates_n_classes, ax, curve_label=added_curve_label
            )
        else:
            _plot_roc(b.fpr, b.tpr, ax=ax, curve_label=added_curve_label)


class ROC(Plot):
    """
    Plot ROC curve
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).

    y_score : array-like, shape = [n_samples] or [n_samples, 2] for binary
        classification or [n_samples, n_classes] for multiclass
        Target scores (estimator predictions).

    fpr : ndarray of shape (>2,), default: None
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= `thresholds[i]`. If
        None, it will be calculated based on y_true and y_score.

    tpr : ndarray of shape (>2,), default: None
        Increasing true positive rates such that element `i` is the true
        positive rate of predictions with score >= `thresholds[i]`. If
        None, it will be calculated based on y_true and y_score.

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
    .. plot:: ../examples/roc_new_api.py

    .. plot:: ../examples/roc_add.py

    Notes
    -----
    .. versionadded:: 0.8.4
    """

    @SKLearnEvaluationLogger.log(feature="plot", action="roc-init")
    def __init__(self, y_true, y_score, fpr=None, tpr=None, ax=None):

        if y_true is not None and y_score is not None:
            warn(
                "ROC will change its signature in version 0.10"
                ", please use ROC.from_raw_data",
                FutureWarning,
                stacklevel=2,
            )

        if ax is None:
            self.figure = plt.figure()
            ax = self.figure.add_subplot()

        # check data shape?
        if tpr is None or fpr is None:

            if any((val is None for val in (y_true, y_score))):
                raise ValueError("y_true and y_score are needed to plot ROC")

            # get the number of classes based on the shape of y_score
            y_score_is_vector = is_column_vector(y_score) or is_row_vector(y_score)
            if y_score_is_vector:
                n_classes = 2
            else:
                _, n_classes = y_score.shape

            if n_classes > 2:
                # convert y_true to binary format
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

                fpr, tpr, _ = _roc_curve_multi(y_true_bin, y_score)
                self.roc_rates_n_classes = []
                for i in range(n_classes):
                    fpr_, tpr_, _ = roc_curve(y_true_bin[:, i], y_score[:, i])

                    d = {"fpr": fpr_.tolist(), "tpr": tpr_.tolist()}
                    self.roc_rates_n_classes.append(d)
            else:
                if y_score_is_vector:
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                else:
                    fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])

        self.fpr = fpr
        self.tpr = tpr
        self.ax = ax

        if hasattr(self, "roc_rates_n_classes"):
            self.ax = _plot_roc_multi_classification(
                self.fpr, self.tpr, self.roc_rates_n_classes, self.ax
            )
        else:
            self.ax = _plot_roc(self.fpr, self.tpr, ax)

    def __sub__(self):
        raise NotImplementedError("Not applicable for ROC")

    @SKLearnEvaluationLogger.log(feature="plot", action="roc-add")
    def __add__(self, other):
        return ROCAdd(self, other)

    def _get_data(self):
        return {
            "class": "sklearn_evaluation.plot.ROC",
            "version": __version__,
            "fpr": self.fpr.tolist(),
            "tpr": self.tpr.tolist(),
        }

    @classmethod
    def from_dump(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        fpr = np.array(data["fpr"])
        tpr = np.array(data["tpr"])

        return cls(y_true=None, y_score=None, fpr=fpr, tpr=tpr, ax=None)

    @classmethod
    def from_raw_data(cls, y_true, y_score):
        return cls(y_true, y_score)
