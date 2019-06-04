"""
Plotting 2 discrete parameters in a heatmap
"""
import numpy as np
from sklearn_evaluation.plot.util import set_default_ax


@set_default_ax
def heatmap(observations, ax=None):
    """
    Plot observations in a heatmap. The data must be a list of dictionaries,
    each dictionary must have a 'params' and 'data' keys. 'params' is another
    dictionary with {param_name: param_value} pairs and data is the number
    to plot in the heatmap
    """
    param_names = sorted(observations[0]['params'].keys())

    params_unique = {val: set([entry['params'][val] for entry in observations])
                     for val in param_names}

    shape = [len(params_unique[param]) for param in param_names]

    val2coord = {param: {v: i for i, v
                         in enumerate(sorted(params_unique[param]))}
                 for param in param_names}

    m = np.empty(shape)
    m[:] = np.nan

    for obs in observations:
        params, data = obs['params'], obs['data']
        i, j = [val2coord[name][params[name]] for name in param_names]
        m[i, j] = data

    ax.imshow(m)
    ax.set_xticks(np.arange(shape[1]))
    ax.set_yticks(np.arange(shape[0]))

    y_labels, x_labels = [sorted(params_unique[name]) for name in param_names]

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel(param_names[1])
    ax.set_ylabel(param_names[0])

    for i in range(shape[1]):
        for j in range(shape[0]):
            ax.text(j, i, m[i, j], ha='center', va='center', color='w')
