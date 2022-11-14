import numpy as np
import matplotlib.pyplot as plt

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


def target_plot(y_train,
                y_test=None,
                labels=None,
                mode='balance',
                color=None,
                colormap=None,
                ax=None):
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
    y_test : array-like, optional
        Array or list of shape (m,) that contains discrete data. If
        specified, the bar chart will be drawn in compare mode.

    labels: list, optional
        A list of class names for the x-axis if the target is already encoded.
        Ensure that the labels are ordered lexicographically with respect to
        the values in the target. A common use case is to pass
        ``LabelEncoder.classes_`` as this parameter. If not specified, the labels
        in the data will be used.

    colors: list of strings
        Specify colors for the barchart (will override colormap if both are provided).
    colormap : string or matplotlib cmap
        Specify a colormap to color the classes.

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

    """

    _validate_target(y_train)
    _validate_target(y_test)
    # Get the unique values from the dataset
    targets = (y_train, ) if y_test is None else (y_train, y_test)
    classes_ = unique_labels(*targets)
    if labels is not None:
        if len(labels) != len(classes_):
            raise ValueError(
                (
                    "discovered {} classes in the data, does not match "
                    "the {} labels specified."
                ).format(len(classes_), len(labels))
            )

    if ax is None:
        ax = plt.gca()
    if mode == 'balance':
        support_ = np.array([(y_train == idx).sum() for idx in classes_])
        ax.bar(
            np.arange(len(support_)),
            support_,
            #colors=colors,
            align="center",
            width=0.5,
        )
    else:
        support_ = np.array(
            [[(y == idx).sum() for idx in classes_] for y in targets]
        )
        bar_width = 0.35
        legends = ["train", "test"]

        for idx, support in enumerate(support_):
            index = np.arange(len(classes_))
            if idx > 0:
                index = index + bar_width

            ax.bar(
                index, support, bar_width,
                #color=colors[idx],
                label=legends[idx]
            )

    ax.set_title("Class Balance for {:,} Instances".format(support_.sum()))

    # Set the x ticks with the class names or labels if specified
    labels = labels if labels else classes_
    xticks = np.arange(len(labels))
    if mode == 'compare':
        xticks = xticks + (0.35 / 2)

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)

    # Compute the ceil ing for the y limit
    cmax = support_.max()
    ax.set_ylim(0, cmax + cmax * 0.1)
    ax.set_ylabel("support")

    # Remove the vertical grid
    ax.grid(False, axis="x")
    if mode == 'compare':
        ax.legend(frameon=True)

    return ax
