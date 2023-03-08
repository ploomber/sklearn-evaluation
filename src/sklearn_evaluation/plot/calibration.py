"""
Calibration curve

NOTE: this is largely based in the scikit-plot implementation. License below.

MIT License

Copyright (c) [2018] [Reiichiro Nakano]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve as sk_calibration_curve
from sklearn.utils import column_or_1d

from sklearn_evaluation import __version__
from sklearn_evaluation.util import isiterofiter
from ploomber_core.exceptions import modify_exceptions
from sklearn_evaluation.telemetry import SKLearnEvaluationLogger
from sklearn_evaluation.plot.plot import AbstractComposedPlot, AbstractPlot
from sklearn_evaluation.plot.style import apply_theme, get_color_palette


@apply_theme()
def _set_ax_settings(ax, name):
    ax.set_title(name)
    ax.set_xlabel("Mean predicted value")
    ax.set_ylabel("Fraction of positives")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="lower right")


@modify_exceptions
def _validate_metrics_input(mean_predicted_value, fraction_of_positives):
    if any((val is None for val in (mean_predicted_value, fraction_of_positives))):
        raise ValueError(
            "Mean_predicted_value and fraction_of_positives"
            " are "
            "needed to plot Calibration Curve"
        )

    # validate input metrics of same length
    if len(mean_predicted_value) != len(fraction_of_positives):
        raise ValueError(
            "mean_predicted_value and fraction_of_positives lengths should correspond. "
            f"Received: mean_predicted_value {len(mean_predicted_value)} "
            f"!= fraction_of_positives {len(fraction_of_positives)}"
        )


@modify_exceptions
def _validate_raw_data(probabilities, y_true, label):
    if any((val is None for val in (probabilities, y_true))):
        raise ValueError(
            "probabilities and y_true" " are " "needed to plot Calibration Curve"
        )

    if not isinstance(probabilities, list):
        raise ValueError("`probabilities` should be a list.")

    if isiterofiter(y_true) and len(y_true) != len(probabilities):
        raise ValueError(
            f"y_true (Received : {len(y_true)}) and probabilities "
            f"(Received : {len(probabilities)}) should have the "
            "same size when y_true is an iterator of array-like objects"
        )
    if len(label) != len(probabilities):
        raise ValueError(
            "Length {} of `label` does not match length {} of"
            " `probabilities`".format(len(label), len(probabilities))
        )


def _plot_from_metrics(mpv, fop, label, color, ax):
    for mpv, fp, name, l in zip(mpv, fop, label, color):
        ax.plot(
            mpv,
            fp,
            "s-",
            label=name,
            color=l,
        )

    return ax


def _generate_colors(cmap, n_color):
    if cmap:
        colors = [plt.cm.get_cmap(cmap)(float(i) / n_color) for i in range(n_color)]
    else:
        colors = get_color_palette()

    return colors


class CalibrationCurve(AbstractPlot):

    """
    Parameters
    ----------
    mean_predicted_value : ndarray of shape (n_bins,) or smaller
        The mean predicted probability in each bin.

    fraction_of_positives : ndarray of shape (n_bins,) or smaller
        The proportion of samples whose class is the positive class, in each
        bin.

    label : list of str, optional
            A list of strings, where each string refers to the name of the
            classifier that produced the corresponding probability estimates in
            `probabilities`. If ``None``, the names "Classifier 1", "Classifier 2",
            etc. will be used.

    cmap : string or :class:`matplotlib.colors.Colormap` instance, optional
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

    Examples
    --------
    .. plot:: ../examples/calibration_curve_oop.py

    .. plot:: ../examples/calibration_curve_diff_sample_size.py

    .. plot:: ../examples/calibration_curve_add.py

    Notes
    -----
    .. versionadded:: 0.11.1
    """

    @modify_exceptions
    @SKLearnEvaluationLogger.log(feature="plot", action="calibration-curve-init")
    def __init__(
        self,
        mean_predicted_value,
        fraction_of_positives,
        label=None,
        cmap=None,
    ):
        self.mean_predicted_value = mean_predicted_value
        self.fraction_of_positives = fraction_of_positives
        self.label = label
        self.cmap = cmap

    def plot(self, ax=None):
        """Create the plot
        Parameters
        -----------
        ax : matplotlib.Axes
            An Axes object to add the plot to
        """
        if ax is None:
            _, ax = plt.subplots()

        _validate_metrics_input(self.mean_predicted_value, self.fraction_of_positives)

        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        color = _generate_colors(self.cmap, len(self.mean_predicted_value))

        if self.label is None:
            self.label = [
                f"Classifier {i+1}" for i in range(len(self.mean_predicted_value))
            ]

        _plot_from_metrics(
            self.mean_predicted_value,
            self.fraction_of_positives,
            self.label,
            color,
            ax,
        )

        _set_ax_settings(ax, "Calibration plots (Reliability Curves)")

        self.ax_ = ax
        self.figure_ = ax.figure

        return self

    @classmethod
    @modify_exceptions
    def from_raw_data(
        cls, y_true, probabilities, *, label=None, n_bins=10, cmap=None
    ):
        """
        Plots calibration curves for a set of classifier probability estimates.
        Calibration curves help determining whether you can interpret predicted
        probabilities as confidence level. For example, if we take a
        well-calibrated and take the instances where the score is 0.8, 80% of those
        instanes should be from the positive class. This function only works for
        binary classifiers.

        Parameters
        ----------
        y_true : array-like, shape = [n_samples] or list with array-like:
            Ground truth (correct) target values. If passed a single array-
            object, it assumes all the `probabilities` have the same shape as
            `y_true`. If passed a list, it expects `y_true[i]` to have the same
            size as `probabilities[i]`
        probabilities : list of array-like, shape (n_samples, 2) or (n_samples,)
            A list containing the outputs of binary classifiers'
            :func:`predict_proba` method or :func:`decision_function` method.
        label : list of str, optional)
            A list of strings, where each string refers to the name of the
            classifier that produced the corresponding probability estimates in
            `probabilities`. If ``None``, the names "Classifier 1", "Classifier 2",
            etc. will be used.
        n_bins : int, optional, default=10
            Number of bins. A bigger number requires more data.
        cmap : string or :class:`matplotlib.colors.Colormap` instance, optional
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html
        """

        _validate_raw_data(probabilities, y_true, label)

        if isiterofiter(y_true):
            classes = np.unique(y_true[0])

        else:
            y_true = len(probabilities) * [y_true]
            classes = np.unique(y_true)

        if len(classes) > 2:
            raise ValueError(
                "plot_calibration_curve only " "works for binary classification"
            )

        if label is None:
            label = ["Classifier {}".format(x + 1) for x in range(len(probabilities))]

        mean_predicted_value = [[] for _ in range(len(probabilities))]
        fraction_of_positives = [[] for _ in range(len(probabilities))]

        for i, (probas, y_true_) in enumerate(zip(probabilities, y_true)):
            probas = np.asarray(probas)
            if probas.ndim > 2:
                raise ValueError(
                    "Index {} in probabilities has invalid "
                    "shape {}".format(i, probas.shape)
                )
            if probas.ndim == 2:
                probas = probas[:, 1]

            if probas.shape != y_true_.shape:
                raise ValueError(
                    "Index {} in probabilities has invalid "
                    "shape {}".format(i, probas.shape)
                )

            (fraction_of_positives[i], mean_predicted_value[i]) = sk_calibration_curve(
                y_true_, probas, n_bins=n_bins
            )

        return cls(
            mean_predicted_value, fraction_of_positives, label=label, cmap=cmap
        ).plot()

    def __add__(self, another):
        return CalibrationCurveAdd(
            mean_predicted_value_list=[
                self.mean_predicted_value,
                another.mean_predicted_value,
            ],
            fraction_of_positives_list=[
                self.fraction_of_positives,
                another.fraction_of_positives,
            ],
            label_list=[self.label, another.label],
            cmaps=[self.cmap, another.cmap],
        ).plot()

    @classmethod
    def _from_data(cls):
        pass

    def _get_data(self):
        return {
            "class": "sklearn_evaluation.plot.calibration.CalibrationCurve",
            "mean_predicted_value": self.mean_predicted_value,
            "fraction_of_positives": self.fraction_of_positives,
            "label": self.label,
            "color": self.color,
            "version": __version__,
        }


class CalibrationCurveAdd(AbstractComposedPlot):
    @modify_exceptions
    @SKLearnEvaluationLogger.log(feature="plot", action="calibration-curve-add-init")
    def __init__(
        self,
        mean_predicted_value_list,
        fraction_of_positives_list,
        label_list,
        cmaps=None,
    ):
        self.mean_predicted_value_list = mean_predicted_value_list
        self.fraction_of_positives_list = fraction_of_positives_list
        self.label_list = label_list
        self.cmaps = cmaps

    def plot(self, ax=None):
        """Generate a new plot with overlapping Calibration curves
        Parameters
        -----------
        ax : matplotlib.Axes
            An Axes object to add the plot to

        """

        if ax is None:
            _, ax = plt.subplots()

        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        plot_one_length = len(self.mean_predicted_value_list[0])
        plot_two_length = len(self.mean_predicted_value_list[1])

        # Consider the first cmap for generating colors on combined plot
        color = _generate_colors(self.cmaps[0], plot_one_length + plot_two_length)

        ax = _plot_from_metrics(
            self.mean_predicted_value_list[0],
            self.fraction_of_positives_list[0],
            self.label_list[0],
            color[:plot_one_length],
            ax,
        )

        ax = _plot_from_metrics(
            self.mean_predicted_value_list[1],
            self.fraction_of_positives_list[1],
            self.label_list[1],
            color[plot_one_length:],
            ax,
        )

        _set_ax_settings(ax, "Calibration Curve Compare")
        self.ax_ = ax
        self.figure_ = ax.figure
        return self


@modify_exceptions
@SKLearnEvaluationLogger.log(feature="plot")
def calibration_curve(
    y_true, probabilities, clf_names=None, n_bins=10, cmap="nipy_spectral", ax=None
):
    """
    Plots calibration curves for a set of classifier probability estimates.
    Calibration curves help determining whether you can interpret predicted
    probabilities as confidence level. For example, if we take a
    well-calibrated and take the instances where the score is 0.8, 80% of those
    instanes should be from the positive class. This function only works for
    binary classifiers.

    Parameters
    ----------

    y_true : array-like, shape = [n_samples] or list with array-like:
        Ground truth (correct) target values. If passed a single array-
        object, it assumes all the `probabilities` have the same shape as
        `y_true`. If passed a list, it expects `y_true[i]` to have the same
        size as `probabilities[i]`
    probabilities : list of array-like, shape (n_samples, 2) or (n_samples,)
        A list containing the outputs of binary classifiers'
        :func:`predict_proba` method or :func:`decision_function` method.
    clf_names : list of str, optional)
        A list of strings, where each string refers to the name of the
        classifier that produced the corresponding probability estimates in
        `probabilities`. If ``None``, the names "Classifier 1", "Classifier 2",
        etc. will be used.
    n_bins : int, optional, default=10
        Number of bins. A bigger number requires more data.
    cmap : string or :class:`matplotlib.colors.Colormap` instance, optional
        Colormap used for plotting the projection. View Matplotlib Colormap
        documentation for available options.
        https://matplotlib.org/users/colormaps.html
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../examples/calibration_curve.py
    """

    return CalibrationCurve.from_raw_data(
        y_true, probabilities, label=clf_names, n_bins=n_bins, cmap=cmap
    ).ax_


@apply_theme()
@modify_exceptions
def scores_distribution(
    y_scores, n_bins=5, title="Predictions distribution", color=None, ax=None
):
    """Generate a histogram from model's predictions

    Parameters
    ----------
    y_scores : array-like, shape=(n_samples, )
        Scores produced by a trained model for a single class
    n_bins : int, default=5
        Number of histogram bins
    title : title of the plot. Defaults to Predictions Distribution
    color : color of the histogram. Defaults to blue.
    ax: matplotlib Axes, default=None
        Axes object to draw the plot onto, otherwise uses current Axes
    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot
    Examples
    --------
    .. plot:: ../examples/scores_distribution.py
    """

    if ax is None:
        _, ax = plt.subplots()

    y_scores = column_or_1d(y_scores)

    # this is how the calibration curve computes the bins, we do it the same
    # way so it matches
    # https://github.com/scikit-learn/scikit-learn/blob/f3f51f9b611bf873bd5836748647221480071a87/sklearn/calibration.py#L989
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ax.hist(y_scores, range=(0, 1), bins=bins, color=color)

    ax.set(
        title=title,
        xlabel="Mean predicted probability",
        ylabel="Count",
    )

    return ax
