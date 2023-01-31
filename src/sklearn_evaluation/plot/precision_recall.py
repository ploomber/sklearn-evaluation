import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn_evaluation.telemetry import SKLearnEvaluationLogger
from sklearn_evaluation.util import is_column_vector, is_row_vector
from sklearn_evaluation.plot.plot import AbstractComposedPlot, AbstractPlot
from sklearn_evaluation import __version__
from ploomber_core.exceptions import modify_exceptions

from warnings import warn  # noqa


_default_label = "Precision Recall Curve"


def _set_ax_settings(ax, name):
    ax.set_title(name)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="best")


@modify_exceptions
def _validate_raw_input(y_true, y_score):
    if any((val is None for val in (y_true, y_score))):
        raise ValueError("y_true and y_score are needed to plot Precision-Recall")


@modify_exceptions
def _validate_metrics_input(precision, recall):
    if any((val is None for val in (precision, recall))):
        raise ValueError("precision and recall are " "needed to plot Precision-Recall")

    # validate input metrics of same length
    if len(precision) != len(recall):
        raise ValueError(
            "precision and recall lengths should correspond. "
            f"Received: precision {len(precision)} != recall {len(recall)}"
        )


def _precision_recall_metrics(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return precision, recall


def _precision_recall_metrics_multiclass(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
    return precision, recall


def _multiclass_metrics_from_raw(y_true, y_score, n_classes, label):
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

    plot_metrics_recall, plot_metrics_precision = [[] for _ in range(n_classes + 1)], [
        [] for _ in range(n_classes + 1)
    ]

    (
        plot_metrics_precision[0],
        plot_metrics_recall[0],
    ) = _precision_recall_metrics_multiclass(y_true_bin, y_score)

    for i in range(n_classes):
        (
            plot_metrics_precision[i + 1],
            plot_metrics_recall[i + 1],
        ) = _precision_recall_metrics(y_true_bin[:, i], y_score[:, i])

    if label is None:
        label = ["Micro-average"] + _generate_labels(n_classes)

    return plot_metrics_precision, plot_metrics_recall, label


def _plot_metrics_multiclass(precision, recall, labels, ax):
    for r, p, l in zip(recall, precision, labels):
        ax.plot(r, p, label=l)
    return ax


def _plot_metrics_binary(precision, recall, label, ax):
    ax.plot(recall, precision, label=label)
    return ax


def _generate_labels(n_classes):
    return [f"PR Curve {i + 1}" for i in range(n_classes)]


class PrecisionRecall(AbstractPlot):
    """
    Plot precision recall curve.

    Parameters
    ----------
    precision : array-like, shape = [n_samples], when task is binary classification,
                or shape = [n_classes, n_samples], when task is multiclass
                classification.

    recall : array-like, shape = [n_samples], when task is binary classification.
             or shape = [n_classes, n_samples], when task is multiclass classification.

    label : string when task is binary classification, optional
            list of strings when task is multiclass classification
            this is used for labelling the curves. Defaults to precision recall.
            Make sure that the order of the labels corresponds to the order in
            which recall/precision arrays are passed to the constructor.

    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Examples
    --------
    .. plot:: ../examples/precision_recall_oop.py

    .. plot:: ../examples/precision_recall_add.py

    Notes
    -----
    .. versionadded:: 0.10.1
    """

    @modify_exceptions
    @SKLearnEvaluationLogger.log(feature="plot", action="precision-recall-init")
    def __init__(self, precision, recall, label=None):
        self.precision = precision
        self.recall = recall
        self.label = label

        _validate_metrics_input(self.precision, self.recall)

    def plot(self, ax=None):
        """Create the plot
        Parameters
        -----------
        ax : matplotlib.Axes
            An Axes object to add the plot to
        """
        if ax is None:
            _, ax = plt.subplots()

        # check if multiclass
        if (
            all(isinstance(elem, (list, np.ndarray)) for elem in self.precision)
            and all(isinstance(elem, (list, np.ndarray)) for elem in self.recall)
            and len(self.precision) > 1
        ):

            if self.label is None:
                self.label = _generate_labels(len(self.recall))

            # Calculate AUC and append to label
            label = []
            for ind, (p, r) in enumerate(zip(self.precision, self.recall)):
                auc_score = auc(r, p)
                label.append(f"{self.label[ind]}, AUC={auc_score:.2f}")

            _plot_metrics_multiclass(
                self.precision,
                self.recall,
                label,
                ax,
            )

        else:
            if self.label is None:
                self.label = _default_label

            # User passes list of list for binary classification
            recall = (
                self.recall[0]
                if isinstance(self.recall[0], (list, np.ndarray))
                else self.recall
            )
            precision = (
                self.precision[0]
                if isinstance(self.precision[0], (list, np.ndarray))
                else self.precision
            )

            auc_score = auc(recall, precision)
            ax = _plot_metrics_binary(
                precision, recall, f"{self.label}, AUC={auc_score:.2f}", ax
            )

        _set_ax_settings(ax, "Precision Recall")
        self.ax_ = ax
        self.figure_ = ax.figure
        return self

    @classmethod
    def _from_data(cls):
        pass

    @SKLearnEvaluationLogger.log(feature="plot", action="precision-recall-add")
    def __add__(self, another):
        return PrecisionRecallAdd(
            precisions=[self.precision, another.precision],
            recalls=[self.recall, another.recall],
            labels=[self.label, another.label],
        ).plot()

    def __sub__(self, another):
        raise NotImplementedError(
            f"{type(self).__name__!r} doesn't support the substract (-) operator"
        )

    @classmethod
    @modify_exceptions
    def from_raw_data(cls, y_true, y_score, *, label=None):
        """
        Plot precision-recall curve from raw data.

        Parameters
        ----------
        y_true : array-like, shape = [n_samples]
            Correct target values (ground truth).
        y_score : array-like, shape = [n_samples] or [n_samples, 2] for binary
                  classification or [n_samples, n_classes] for multiclass
            Target scores (estimator predictions).
        label : string or list, optional
            labels for the curves

        Notes
        -----
        It is assumed that the y_score parameter columns are in order. For example,
        if ``y_true = [2, 2, 1, 0, 0, 1, 2]``, then the first column in y_score
        must contain the scores for class 0, second column for class 1 and so on.

        """

        _validate_raw_input(y_true, y_score)

        # get the number of classes from y_score
        # vector when y_score is probability estimates of the positive
        # class. Else it is non-thresholded measure of decisions.
        y_score_is_vector = is_column_vector(y_score) or is_row_vector(y_score)
        if y_score_is_vector:
            n_classes = 2
        else:
            _, n_classes = y_score.shape

        if n_classes > 2:
            if label is not None:
                label = ["Micro-average"] + label
            # convert y_true to binary format
            precision, recall, label = _multiclass_metrics_from_raw(
                y_true, y_score, n_classes, label
            )

            return cls(
                precision,
                recall,
                label=label,
            ).plot()

        else:
            precision, recall = (
                _precision_recall_metrics(y_true, y_score)
                if y_score_is_vector
                else _precision_recall_metrics(y_true, y_score[:, 1])
            )

            return cls(precision, recall, label=label).plot()

    def _get_data(self):
        return {
            "class": "sklearn_evaluation.plot.precision_recall.PrecisionRecall",
            "recall": list(self.recall),
            "precision": list(self.precision),
            "label": self.label,
            "version": __version__,
        }


class PrecisionRecallAdd(AbstractComposedPlot):
    def __init__(self, precisions, recalls, labels) -> None:
        self.precisions = precisions
        self.recalls = recalls
        self.labels = labels

    def plot(self, ax=None):
        """
        Create the plot

        Parameters
        -----------
        ax : matplotlib.Axes
            An Axes object to add the plot to
        """

        if ax is None:
            _, ax = plt.subplots()

        for model_ind, (p, r, l) in enumerate(
            zip(*(self.precisions, self.recalls, self.labels))
        ):
            # check if multiclass
            if all(isinstance(elem, (list, np.ndarray)) for elem in r) and all(
                isinstance(elem, (list, np.ndarray)) for elem in p
            ):
                label = []
                for ind, (p_item, r_item) in enumerate(zip(p, r)):
                    label.append(f"{l[ind]}, AUC={auc(r_item, p_item):.2f}")

                _plot_metrics_multiclass(p, r, label, ax)

            else:
                # User passes list of list for binary
                r = r[0] if isinstance(r[0], (list, np.ndarray)) else r
                p = p[0] if isinstance(p[0], (list, np.ndarray)) else p
                label = f"{l}, AUC={auc(r,p):.2f}"
                ax = _plot_metrics_binary(p, r, label, ax)

        _set_ax_settings(ax, "Precision Recall compare")
        self.ax_ = ax
        self.figure_ = ax.figure
        return self


@modify_exceptions
@SKLearnEvaluationLogger.log(feature="plot")
def precision_recall(y_true, y_score, ax=None):

    """
    Plot precision-recall curve.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_samples] or [n_samples, 2] for binary
              classification or [n_samples, n_classes] for multiclass
              Target scores (estimator predictions).
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Notes
    -----
    It is assumed that the y_score parameter columns are in order. For example,
    if ``y_true = [2, 2, 1, 0, 0, 1, 2]``, then the first column in y_score
    must contain the scores for class 0, second column for class 1 and so on.

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../examples/precision_recall.py
    """

    return PrecisionRecall.from_raw_data(y_true, y_score).ax_
