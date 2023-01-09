"""
KS statistics plot

NOTE: this is largely based in the scikit-plot metrics module. License below.

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
from sklearn.preprocessing import LabelEncoder
from sklearn_evaluation.telemetry import SKLearnEvaluationLogger
from ploomber_core.exceptions import modify_exceptions


def _binary_ks_curve(y_true, y_score):
    """This function generates the points necessary to calculate the KS
    Statistic curve.

    Parameters
    ----------

    y_true (array-like, shape (n_samples)): True labels of the data.
            Refer: https://numpy.org/doc/stable/glossary.html#term-array-like

    y_score (array-like, shape (n_samples)): Probability predictions of
            the positive class.
            Refer: https://numpy.org/doc/stable/glossary.html#term-array-like

    Returns
    -------
    thresholds (numpy.ndarray): An array containing the X-axis values for
        plotting the KS Statistic plot.
    pct1 (numpy.ndarray): An array containing the Y-axis values for one
        curve of the KS Statistic plot.
    pct2 (numpy.ndarray): An array containing the Y-axis values for one
        curve of the KS Statistic plot.
    ks_statistic (float): The KS Statistic, or the maximum vertical
        distance between the two curves.
    max_distance_at (float): The X-axis value at which the maximum vertical
        distance between the two curves is seen.
    classes (np.ndarray, shape (2)): An array containing the labels of the
        two classes making up `y_true`.

    Raises
    -------
    ValueError: If `y_true` is not composed of 2 classes. The KS Statistic
        is only relevant in binary classification.
    """

    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    lb = LabelEncoder()
    encoded_labels = lb.fit_transform(y_true)
    if len(lb.classes_) != 2:
        raise ValueError(
            "Cannot calculate KS statistic for data with "
            "{} category/ies".format(len(lb.classes_))
        )
    idx = encoded_labels == 0
    data1 = np.sort(y_score[idx])
    data2 = np.sort(y_score[np.logical_not(idx)])

    ctr1, ctr2 = 0, 0
    thresholds, pct1, pct2 = [], [], []
    while ctr1 < len(data1) or ctr2 < len(data2):

        # Check if data1 has no more elements
        if ctr1 >= len(data1):
            current = data2[ctr2]
            while ctr2 < len(data2) and current == data2[ctr2]:
                ctr2 += 1

        # Check if data2 has no more elements
        elif ctr2 >= len(data2):
            current = data1[ctr1]
            while ctr1 < len(data1) and current == data1[ctr1]:
                ctr1 += 1

        else:
            if data1[ctr1] > data2[ctr2]:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1

            elif data1[ctr1] < data2[ctr2]:
                current = data1[ctr1]
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

            else:
                current = data2[ctr2]
                while ctr2 < len(data2) and current == data2[ctr2]:
                    ctr2 += 1
                while ctr1 < len(data1) and current == data1[ctr1]:
                    ctr1 += 1

        thresholds.append(current)
        pct1.append(ctr1)
        pct2.append(ctr2)

    thresholds = np.asarray(thresholds)
    pct1 = np.asarray(pct1) / float(len(data1))
    pct2 = np.asarray(pct2) / float(len(data2))

    if thresholds[0] != 0:
        thresholds = np.insert(thresholds, 0, [0.0])
        pct1 = np.insert(pct1, 0, [0.0])
        pct2 = np.insert(pct2, 0, [0.0])
    if thresholds[-1] != 1:
        thresholds = np.append(thresholds, [1.0])
        pct1 = np.append(pct1, [1.0])
        pct2 = np.append(pct2, [1.0])

    differences = pct1 - pct2
    ks_statistic, max_distance_at = (
        np.max(differences),
        thresholds[np.argmax(differences)],
    )

    return thresholds, pct1, pct2, ks_statistic, max_distance_at, lb.classes_


@SKLearnEvaluationLogger.log(feature="plot")
@modify_exceptions
def ks_statistic(
    y_true,
    y_score,
    figsize=None,
    title_fontsize="large",
    text_fontsize="medium",
    ax=None,
):
    """Generates the KS Statistic plot from labels and scores/probabilities

    Parameters
    ----------

    y_true : array-like, shape=[n_samples,]
        Ground truth target values.
        Refer: https://numpy.org/doc/stable/glossary.html#term-array-like

    y_score : array-like, shape=[n_samples, n_classes]
        Prediction probabilities for each class returned by a classifier.
        Refer: https://numpy.org/doc/stable/glossary.html#term-array-like

    figsize : 2-tuple, optional
        Tuple denoting figure size of the plot e.g. (6, 6). Defaults to ``None``.

    title_fontsize : string or int, optional
        Matplotlib-style fontsizes. Use e.g. "small", "medium", "large"
        or integer-values. Defaults to "large".

    text_fontsize : string or int, optional
        Matplotlib-style fontsizes. Use e.g. "small", "medium", "large"
        or integer-values. Defaults to "medium".

    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../examples/ks_statistics.py

    Notes
    -----
    .. versionadded:: 0.8.4

    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError(
            "Cannot calculate KS statistic for data with "
            "{} category/ies".format(len(classes))
        )
    probas = y_score

    # Compute KS Statistic curves
    thresholds, pct1, pct2, ks_statistic, max_distance_at, classes = _binary_ks_curve(
        y_true, probas[:, 1].ravel()
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title("KS Statistics Plot", fontsize=title_fontsize)

    ax.plot(thresholds, pct1, lw=3, label="Class {}".format(classes[0]))
    ax.plot(thresholds, pct2, lw=3, label="Class {}".format(classes[1]))
    idx = np.where(thresholds == max_distance_at)[0][0]
    ax.axvline(
        max_distance_at,
        *sorted([pct1[idx], pct2[idx]]),
        label="KS Statistic: {:.3f} at {:.3f}".format(ks_statistic, max_distance_at),
        linestyle=":",
        lw=3,
        color="black"
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.set_xlabel("Threshold", fontsize=text_fontsize)
    ax.set_ylabel("Percentage below threshold", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc="lower right", fontsize=text_fontsize)

    return ax
