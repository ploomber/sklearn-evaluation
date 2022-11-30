"""
Plot for target analysis

NOTE: this is based on the yellowbricks target analysis module. License below.

Copyright 2016-2020 The scikit-yb developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import matplotlib.pyplot as plt
from ..telemetry import SKLearnEvaluationLogger
from sklearn.utils.multiclass import unique_labels, type_of_target


def _validate_target(y):
    """
        Raises a value error if the target is not a classification target.
        """
    # Ignore None values
    if y is None:
        return

    y_type = type_of_target(y)
    if y_type not in ("binary", "multiclass"):
        raise TypeError(
            "'{}' target type not supported, only binary and multiclass".
            format(y_type))


@SKLearnEvaluationLogger.log(feature='plot')
def target_analysis(y_train, y_test=None, labels=None, colors=None, ax=None):
    """Target analysis plot for visualising class imbalance.

    There are two modes:

    1. Balance mode: if only y_train is specified
    2. Compare mode: if both train and test are specified

    In balance mode, the bar chart is displayed with each class as its own
    color. In compare mode, a side-by-side bar chart is displayed colored
    by train or test respectively.

    Parameters
    ----------
    y_train : array-like
        Array or list of shape (n,) that contains discrete data.
        Refer https://numpy.org/doc/stable/glossary.html#term-array-like
    y_test : array-like, optional
        Array or list of shape (m,) that contains discrete data. If
        specified, the bar chart will be drawn in compare mode.
        Refer https://numpy.org/doc/stable/glossary.html#term-array-like

    labels: list, optional
        A list of class names for the x-axis if the target is already encoded.
        Ensure that the labels are ordered lexicographically with respect to
        the values in the target. A common use case is to pass
        ``LabelEncoder.classes_`` as this parameter. If not specified, the labels
        in the data will be used.

    colors: list of strings
        Specify colors for the barchart.

    ax : :class:`matplotlib.axes.Axes`, optional
        The axes upon which to plot the curve. If None, the plot is drawn
        on the current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/target_analysis.py

    Notes
    -----
    .. versionadded:: 0.8.3

    """

    _validate_target(y_train)
    _validate_target(y_test)
    # Get the unique values from the dataset
    targets = (y_train, ) if y_test is None else (y_train, y_test)
    classes_ = unique_labels(*targets)
    if labels is not None:
        if len(labels) != len(classes_):
            raise ValueError(
                ("Discovered {} classes in the data, does not match "
                 "the {} labels specified.").format(len(classes_),
                                                    len(labels)))

    if ax is None:
        ax = plt.gca()
    mode = 'balance' if y_test is None else 'compare'
    if mode == 'balance':
        support_ = np.array([(y_train == idx).sum() for idx in classes_])
        ax.bar(
            np.arange(len(support_)),
            support_,
            color=colors if colors else "#0070FF",
            align="center",
            width=0.5,
        )
    else:
        support_ = np.array([[(y == idx).sum() for idx in classes_]
                             for y in targets])
        bar_width = 0.35
        legends = ["train", "test"]
        colors = colors if colors else ['#0070FF', '#FF9B00']
        for idx, support in enumerate(support_):
            index = np.arange(len(classes_))
            if idx > 0:
                index = index + bar_width

            ax.bar(index,
                   support,
                   bar_width,
                   color=colors[idx],
                   label=legends[idx])

    ax.set_title("Class Balance for {:,} Instances".format(support_.sum()))

    # Set the x ticks with the class names or labels if specified
    labels = labels if labels else classes_
    xticks = np.arange(len(labels))
    if mode == 'compare':
        xticks = xticks + (0.35 / 2)

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)

    # Compute the ceiling for the y limit
    cmax = support_.max()
    ax.set_ylim(0, cmax + cmax * 0.1)
    ax.set_ylabel("support")

    # Remove the vertical grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#808080')
    ax.autoscale(enable=True)

    if mode == 'compare':
        ax.legend(frameon=True)

    return ax
