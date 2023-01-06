"""
Example implemention of ``AbstractPlot``, and ``ComposedAbstractPlot``. This is
intended to guide developers add new plots, and not intended for new users.

Useful links:

.. plot:: documentation https://matplotlib.org/3.1.3/devel/plot_directive.html
"""
from collections import Counter


import numpy as np
import matplotlib.pyplot as plt

from sklearn_evaluation import __version__
from sklearn_evaluation.plot.plot import AbstractComposedPlot, AbstractPlot


class MyBar(AbstractPlot):
    """
    Bar plot. This is an internal plot targeted for developers, not intended for
    end-users.

    Parameters
    ----------
    count : dict
        A dictionary whose keys are labels and values are counts

    color : string, default=None
        Color for the bars, must be a valid matplotlib color

    name : string, default=None
        A value to identify this plot

    Notes
    -----
    .. versionadded:: 0.9


    Examples
    --------

    Create plot:

    .. plot::

        from sklearn_evaluation.plot._example import MyBar
        MyBar.from_raw_data(["banana", "banana", "apple", "pineapple", "apple"],
                                color="lightblue")

    Compare plots:


    .. plot::

        from sklearn_evaluation.plot._example import MyBar
        one = MyBar.from_raw_data(["banana", "banana", "apple", "pineapple", "apple"])
        another = MyBar.from_raw_data(["banana", "apple",  "pineapple"])
        one + another

    """

    def __init__(self, count, *, color=None, name=None):
        self.count = count
        self.color = color
        self.name = name

    # ax must be optional
    def plot(self, ax=None):
        """Create the plot

        Parameters
        -----------
        ax : matplotlib.Axes
            An Axes object to add the plot to
        """
        if ax is None:
            _, ax = plt.subplots()

        ax.bar(
            list(range(len(self.count))), list(self.count.values()), color=self.color
        )

        # properly label axis and ticks
        ax.set_xticks(list(range(len(self.count))), self.count.keys())

        ax.set_ylabel("Count")
        label = "" if self.name is None else f" ({self.name})"
        ax.set_title("Bar plot" + label)

        # remember to set these attributes
        self.ax_ = ax
        self.figure_ = ax.figure

        # remember to return self
        return self

    @classmethod
    def from_raw_data(cls, things_to_count, *, color=None, name=None):
        """
        check typical naming: such as y_pred, y_score, y_true

        Parameters
        ----------
        things_to_count : list
            The list of elements to count

        color : string, default=None
            Color for the bars, must be a valid matplotlib color

        name : string, default=None
            A value to identify this plot
        """
        count = Counter(things_to_count)
        # remember to call plot here
        return cls(count, color=color, name=name).plot()

    def __add__(self, another):
        return MyBarAdd(
            counts=[self.count, another.count], names=[self.name, another.name]
        ).plot()

    def __sub__(self, another):
        return MyBarSub(counts=[self.count, another.count]).plot()

    @classmethod
    def _from_data(cls):
        pass

    def _get_data(self):
        return {
            "class": "sklearn_evaluation.plot._example.MyBarPlot",
            # ensure all objects are JSON serializable! (some of them might be
            # np.arrays)
            "count": dict(self.count),
            "color": self.color,
            "name": self.name,
            "version": __version__,
        }


class MyBarSub(AbstractComposedPlot):
    def __init__(self, counts, color=None) -> None:
        self.counts = counts
        self.color = color

    def plot(self, ax=None):
        # validate keys are the same
        diff = dict()

        for key, value in self.counts[0].items():
            diff[key] = value - self.counts[1][key]

        if ax is None:
            _, ax = plt.subplots()

        ax.bar(list(range(len(diff))), list(diff.values()), color=self.color)
        ax.set_xticks(list(range(len(diff))), diff.keys())

        # remember to set these attributes
        self.ax_ = ax
        self.figure_ = ax.figure

        ax.set_ylabel("Count")

        # remember to return self
        return self


class MyBarAdd(AbstractComposedPlot):
    def __init__(self, counts, names) -> None:
        self.counts = counts
        self.names = names

        # in a real implementation, input data validation should be performed here

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()

        labels = list(self.counts[0].keys())

        x = np.arange(len(labels))
        width = 0.35

        name_one = self.names[0] or "first"
        name_two = self.names[1] or "second"

        rects1 = ax.bar(
            x - width / 2,
            self.counts[0].values(),
            width,
            label=name_one,
        )
        rects2 = ax.bar(
            x + width / 2,
            self.counts[1].values(),
            width,
            label=name_two,
        )

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("Count")
        ax.set_title(f"Bar plot ({name_one} and {name_two})")
        ax.set_xticks(x, labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        # remember to set these attributes
        self.ax_ = ax
        self.figure_ = ax.figure

        return self


def my_bar(things_to_count, ax=None, color=None):
    """
    Parameters
    ----------
    things_to_count : list
        The list of elements to count

    color : string, default=None
        Color for the bars, must be a valid matplotlib color

    Examples
    --------
    .. plot::

        from sklearn_evaluation.plot._example import my_bar
        my_bar(["banana", "banana", "apple", "pineapple", "apple"],
               color="lightblue")
    """
    # do not call plot here, from_raw_data already calls it!
    return MyBar.from_raw_data(things_to_count=things_to_count, color=color).ax_
