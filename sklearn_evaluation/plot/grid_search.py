from functools import reduce
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from six import string_types

from ..util import (_group_by, _tuple_getter, _mapping_to_tuple_pairs,
                    default_heatmap)


def grid_search(grid_scores, to_vary, to_keep=None, ax=None, kind='line',
                cmap=None):
    """
    Plot results from a sklearnn Grid Search by changing two parameters at most
    and leaving the rest constant.

    Parameters
    ----------
    grid_scores : list of named tuples
        Results from a sklearn Grid Search (get them using the
        grid_scores_ parameter)
    to_vary : str or iterable with len<=2
        Parameter to vary
    to_keep : dictionary-like
        parameter-value pairs, such values will keep fixed to plot. You can
        specify more than one value per parameter
        (e.g. {'n_estimartors': [1, 10]})
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes
    kind : ['line', 'bar']
        This only applies whe to_vary is a single parameter. Changes the
        type of plot
    cmap : matplotlib Colormap
        This only applies when to_vary are two parameters. Colormap used for
        the matrix. If None uses a modified version of matplotlib's OrRd
        colormap.

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = default_heatmap()

    if isinstance(to_vary, string_types) or len(to_vary) == 1:
        return _grid_search_single(grid_scores, to_vary, to_keep, ax, kind)
    elif len(to_vary) == 2:
        return _grid_search_double(grid_scores, to_vary, to_keep, ax, cmap)
    else:
        raise ValueError('to_vary must have length 1 or 2 or be a string')


def _grid_search_single(grid_scores, to_vary, to_keep, ax, kind):
    # get the parameters list
    params_set = list(grid_scores[0].parameters.keys())

    # check how many unique values does to_vary has
    to_vary_unique = len(set([g.parameters[to_vary] for g in grid_scores]))

    # check that the union to_vary and to_keep contains all the values
    # in the params_set

    # remove parameter to vary from the list
    params_set.remove(to_vary)

    # from the parameters that remain, group
    # the scores based on the different combinations
    groups = _group_by(grid_scores, _tuple_getter(params_set))

    # get the elements in the group that the user wants
    if to_keep:
        keys = _mapping_to_tuple_pairs(to_keep)
        groups = {k: v for k, v in groups.items() if k in keys}

    # bar shifter is just a wrapper arounf matplotlib bar chart
    # which automatically computes the position of the bars
    # you need to specify how many groups of bars are gonna be
    # and the size of such groups. The frst time you call it,
    # will plot the first element for every group, then the second one
    # and so on
    if kind == 'bar':
        bar_shifter = BarShifter(g_number=to_vary_unique, g_size=len(groups),
                                 ax=ax)

    for to_keep, group in groups.items():
        # get the x and y coordinates for the plot
        x = [element.parameters[to_vary] for element in group]
        y = [element.mean_validation_score for element in group]
        stds = [element.cv_validation_scores.std() for element in group]


        label = reduce(lambda x, y: x+', '+y, ['{}: {}'.format(*t) for t in to_keep])

        if kind == 'bar':
            bar_shifter(y, yerr=stds, label=label)
        elif kind == 'line':
            is_categorical = isinstance(x[0], string_types)
            if is_categorical:
                ints = range(len(x))
                ax.errorbar(ints, y, yerr=stds, label=label)
                ax.set_xticks(ints)
            else:
                ax.errorbar(x, y, yerr=stds, label=label)
                ax.set_xticks(x)

    ax.set_xticklabels(x)
    ax.set_title('Grid search results')
    ax.set_ylabel('Mean score')
    ax.set_xlabel(to_vary)
    ax.legend(loc="best")
    ax.margins(0.05)
    return ax


def _grid_search_double(grid_scores, to_vary, to_keep, ax, cmap):
    groups = _group_by(grid_scores, _tuple_getter(to_keep.keys()))
    keys = _mapping_to_tuple_pairs(to_keep)
    groups = {k: v for k, v in groups.items() if k in keys}

    # there should be just one group at this point
    if len(groups) > 1:
        raise ValueError('You did not specify x')

    group = list(groups.values())[0]

    # group by every possible combination in to_vary
    matrix_elements = _group_by(group, _tuple_getter(to_vary))

    for v in matrix_elements.values():
        if len(v) > 1:
            raise ValueError('Group with more than one element')

    # on each group there must be only one element, get it
    matrix_elements = {k: v[0] for k, v in matrix_elements.items()}

    # get the unique values for each element
    # and sort the results to make sure the matrix axis
    # is showed in increasing order
    row_names = sorted(set([t[0] for t in matrix_elements.keys()]),
                       key=itemgetter(1))
    col_names = sorted(set([t[1] for t in matrix_elements.keys()]),
                       key=itemgetter(1))

    # size of the matrix
    cols = len(col_names)
    rows = len(row_names)

    # map values to coordinates to populate the marix
    x_coord = {k: v for k, v in zip(col_names, range(cols))}
    y_coord = {k: v for k, v in zip(row_names, range(rows))}

    # replace keys in matrix_elements with their corresponding indexes
    # improve variable naming
    m = {(x_coord[k[1]], y_coord[k[0]]): v for k, v in matrix_elements.items()}

    matrix = np.zeros((rows, cols))

    for (j, i), v in m.items():
        matrix[i][j] = v.mean_validation_score

    # ticks for the axis
    row_labels = ['{}={}'.format(*x) for x in row_names]
    col_labels = ['{}={}'.format(*y) for y in col_names]

    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap)

    # set text on cells
    for (x, y), v in m.items():
        label = '{:.3}'.format(v.mean_validation_score)
        ax.text(x, y, label, horizontalalignment='center',
                verticalalignment='center')

    ax.set_xticks(range(cols))
    ax.set_xticklabels(col_labels, rotation=45)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(row_labels)
    plt.colorbar(im, ax=ax)
    ax.get_figure().tight_layout()
    return ax


class BarShifter:
    def __init__(self, g_number, g_size, ax, scale=0.8):
        self.g_number = g_number
        self.g_size = g_size
        self.ax = ax
        self.i = 0
        self.width = (1.0/g_size)*scale
        self.colors = plt.get_cmap()(np.linspace(0, 1, self.g_size))

    def __call__(self, height, bottom=None, hold=None, **kwargs):
        left = [x+self.i*self.width for x in range(self.g_number)]
        self.ax.bar(left, height, self.width, color=self.colors[self.i],
                    ecolor=self.colors[self.i],
                    **kwargs)
        self.i += 1
        if self.i == self.g_size:
            ticks_pos = [x+(self.width*self.g_size)/2.0 for x in range(self.g_number)]
            self.ax.set_xticks(ticks_pos)
