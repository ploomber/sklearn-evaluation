"""
Plotting 2 discrete parameters in a heatmap
"""
import numpy as np
from sklearn_evaluation.plot.util import set_default_ax
from sklearn_evaluation.plot.matplotlib.data_grid import DataGrid


@set_default_ax
def heatmap(records, ax=None, get_value=lambda data: data,
            get_text=lambda data: data,
            kwargs_text=dict(ha='center', va='center', color='w')):
    """
    """
    dg = DataGrid(records)
    shape = dg.shape

    m = np.empty(shape)
    m[:] = np.nan

    for i, j, data in dg.celliter():
        m[i, j] = get_value(data)

    ax.imshow(m)
    ax.set_xticks(np.arange(shape[1]))
    ax.set_yticks(np.arange(shape[0]))

    ax.set_xticklabels(dg.colnames())
    ax.set_yticklabels(dg.rownames())

    ax.set_xlabel(dg.params[1])
    ax.set_ylabel(dg.params[0])

    for i, j, data in dg.celliter():
        ax.text(j, i, get_text(data), **kwargs_text)

    return ax
