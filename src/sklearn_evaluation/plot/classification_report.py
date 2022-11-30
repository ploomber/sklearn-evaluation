from warnings import warn
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report as sk_classification_report
from matplotlib.figure import Figure

from sklearn_evaluation.plot.classification import _add_values_to_matrix
from sklearn_evaluation.util import default_heatmap
from sklearn_evaluation.plot.plot import Plot
from sklearn_evaluation.plot import _matrix
from sklearn_evaluation import __version__


def _classification_report_add(first, second, keys, target_names, ax):
    _matrix.add(first, second, ax, invert_axis=True, max_=1.0)

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys)

    tick_marks = np.arange(len(target_names))
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(target_names)

    ax.set(title="Classification report (compare)", xlabel="Metric", ylabel="Class")


class ClassificationReportSub(Plot):
    def __init__(self, matrix, matrix_another, keys, target_names) -> None:
        self.figure = Figure()
        ax = self.figure.add_subplot()
        _classification_report_plot(matrix - matrix_another, keys, target_names, ax)
        ax.set(title="Classification report (difference)")


class ClassificationReportAdd(Plot):
    def __init__(self, matrix, matrix_another, keys, target_names) -> None:
        self.figure = Figure()
        self.ax = self.figure.add_subplot()
        _classification_report_add(matrix, matrix_another, keys, target_names, self.ax)


class ClassificationReport(Plot):
    """

    Examples
    --------
    .. plot:: ../../examples/ClassificationReport.py
    """

    def __init__(
        self,
        y_true,
        y_pred,
        *,
        target_names=None,
        sample_weight=None,
        zero_division=0,
        matrix=None,
        keys=None
    ):
        if y_true is not None and matrix is None:
            warn(
                "ClassificationReport will change its signature in version 0.10"
                ", please use ClassificationReport.from_raw_data",
                FutureWarning,
                stacklevel=2,
            )

        self.figure = Figure()
        ax = self.figure.add_subplot()

        if matrix is not None and matrix is not False:
            self.matrix = matrix
            self.keys = keys
            self.target_names = target_names
        else:
            self.matrix, self.keys, self.target_names = _classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                sample_weight=sample_weight,
                zero_division=zero_division,
            )

        _classification_report_plot(self.matrix, self.keys, self.target_names, ax)

    def __sub__(self, other):
        return ClassificationReportSub(
            self.matrix, other.matrix, self.keys, target_names=self.target_names
        )

    def __add__(self, other):
        return ClassificationReportAdd(
            self.matrix, other.matrix, keys=self.keys, target_names=self.target_names
        )

    def _get_data(self):
        return {
            "class": "sklearn_evaluation.plot.ClassificationReport",
            "matrix": self.matrix.tolist(),
            "keys": self.keys,
            "target_names": self.target_names,
            "version": __version__,
        }

    @classmethod
    def from_dump(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            matrix=np.array(data["matrix"]),
            keys=data["keys"],
            target_names=data["target_names"],
            y_true=None,
            y_pred=None,
        )

    @classmethod
    def from_raw_data(
        cls, y_true, y_pred, *, target_names=None, sample_weight=None, zero_division=0
    ):
        # pass matrix=False so we don't emit the future warning
        return cls(
            y_true,
            y_pred,
            target_names=target_names,
            sample_weight=sample_weight,
            zero_division=zero_division,
            matrix=False,
            keys=False,
        )

    @classmethod
    def _from_data(cls, target_names, matrix, keys):
        return cls(
            y_true=None,
            y_pred=None,
            target_names=target_names,
            matrix=np.array(matrix),
            keys=keys,
        )


def _classification_report(
    y_true, y_pred, *, target_names=None, sample_weight=None, zero_division=0
):

    report = sk_classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        sample_weight=sample_weight,
        zero_division=zero_division,
        output_dict=True,
    )

    report = {k: v for k, v in report.items() if "avg" not in k and k != "accuracy"}
    n_classes = len(report.keys())

    target_names = target_names or [str(i) for i in range(n_classes)]

    keys = list(report[target_names[0]].keys())
    rows = [list(row.values()) for row in report.values()]
    matrix = np.array(rows)

    return matrix, keys, target_names


def _classification_report_plot(matrix, keys, target_names, ax):
    _add_values_to_matrix(matrix, ax)

    ax.imshow(matrix, interpolation="nearest", cmap=default_heatmap())

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys)

    tick_marks = np.arange(len(target_names))
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(target_names)

    ax.set(title="Classification report", xlabel="Metric", ylabel="Class")

    return ax


# TODO: add unit test
def classification_report(
    y_true, y_pred, *, target_names=None, sample_weight=None, zero_division=0, ax=None
):
    """Classification report

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth)

    y_pred : array-like, shape = [n_samples]
        Target predicted classes (estimator predictions)

    target_names : list
        List containing the names of the target classes. List must be in order
        e.g. ``['Label for class 0', 'Label for class 1']``. If ``None``,
        generic labels will be generated e.g. ``['Class 0', 'Class 1']``

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    zero_division : bool,  0 or 1
        Sets the value to return when there is a zero division.

    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/classification_report.py

    .. plot:: ../../examples/classification_report_multiclass.py
    """

    if ax is None:
        ax = plt.gca()

    matrix, keys, target_names = _classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )

    return _classification_report_plot(matrix, keys, target_names, ax)
