import numpy as np
import matplotlib.pyplot as plt


def plot(values, orientation, labels=None, sort=True, error=None, ax=None):
    if orientation == 'horizontal':
        return horizontal(values, labels, sort, error, ax)
    else:
        return vertical(values, labels, sort, error, ax)


def horizontal(values, labels=None, sort=True, error=None, ax=None):
    """Horizontal bar plot

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn_evaluation.plot.matplotlib import bar
    >>> values = np.arange(10)
    >>> ax = bar.horizontal(values)
    >>> plt.show()

    Notes
    -----
    https://matplotlib.org/gallery/lines_bars_and_markers/barh.html

    """
    if sort:
        idx_sorted = np.argsort(values)
        values = values[idx_sorted]

        if labels is not None:
            labels = labels[idx_sorted]

    if ax is None:
        ax = plt.gca()

    y_pos = np.arange(len(values))

    if error is None:
        ax.barh(y_pos, values)
    else:
        ax.barh(y_pos, values, xerr=error)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_pos if labels is None else labels)
    ax.invert_yaxis()
    ax.set_ylim([-1, len(values)])
    ax.grid(axis='x')
    # ax.figure.tight_layout()

    return ax


def vertical(values, labels=None, sort=True, error=None, ax=None):
    """Vertical bar plot

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn_evaluation.plot.matplotlib import bar
    >>> values = np.arange(10)
    >>> ax = bar.vertical(values)
    >>> plt.show()

    """
    if sort:
        idx_sorted = np.argsort(values)[::-1]
        values = values[idx_sorted]

        if labels is not None:
            labels = labels[idx_sorted]

    if ax is None:
        ax = plt.gca()

    x_pos = np.arange(len(values))

    if error is None:
        ax.bar(x_pos, values)
    else:
        ax.bar(x_pos, values, yerr=error)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_pos if labels is None else labels)
    ax.set_xlim([-1, len(values)])
    ax.grid(axis='y')
    # ax.figure.tight_layout()

    return ax
