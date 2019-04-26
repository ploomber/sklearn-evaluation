import numpy as np
import matplotlib.pyplot as plt


def horizontal(values, labels=None, error=None):
    """Horizontal bar plot

    Examples
    --------
    >>> from sklearn_evaluation.plot.matplotlib import bar
    >>> values = np.random.rand(10)
    >>> bar.horizontal(values)
    >>> plt.show()

    Notes
    -----
    https://matplotlib.org/gallery/lines_bars_and_markers/barh.html

    """
    y_pos = np.arange(len(values))
    ax = plt.gca()

    if error is None:
        ax.barh(y_pos, values)
    else:
        ax.barh(y_pos, values, xerr=error)

    ax.set_yticks(y_pos if labels is None else labels)
    ax.set_yticklabels(y_pos)
    ax.invert_yaxis()
    return ax
