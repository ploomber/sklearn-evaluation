"""
Plotting 2 discrete parameters in a heatmap
"""
import numpy as np
from sklearn_evaluation.plot.util import set_default_ax
from operator import itemgetter


@set_default_ax
def heatmap(observations, ax=None, get_params=itemgetter('params'),
            get_value=itemgetter('value'), get_text=itemgetter('value'),
            kwargs_text=dict(ha='center', va='center', color='w')):
    """
    Plot observations in a heatmap. The data must be a list of dictionaries,
    each dictionary must have a 'params' and 'data' keys. 'params' is another
    dictionary with {param_name: param_value} pairs and data is the number
    to plot in the heatmap.

    If observations are another kind of object, you can pass callables
    in get_params and get_value to specify how parameters and the value
    to plot are extracted from each observation, if get_text is not None,
    it will be used to plot some text on each cell
    """
    param_names = sorted(get_params(observations[0]).keys())

    params_unique = {val: set([get_params(entry)[val]
                     for entry in observations])
                     for val in param_names}

    shape = [len(params_unique[param]) for param in param_names]

    val2coord = {param: {v: i for i, v
                         in enumerate(sorted(params_unique[param]))}
                 for param in param_names}

    m = np.empty(shape)
    m[:] = np.nan

    for obs in observations:
        params, value = get_params(obs), get_value(obs)
        i, j = [val2coord[name][params[name]] for name in param_names]
        m[i, j] = value

    ax.imshow(m)
    ax.set_xticks(np.arange(shape[1]))
    ax.set_yticks(np.arange(shape[0]))

    y_labels, x_labels = [sorted(params_unique[name]) for name in param_names]

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel(param_names[1])
    ax.set_ylabel(param_names[0])

    for obs in observations:
        params, text = get_params(obs), get_text(obs)
        i, j = [val2coord[name][params[name]] for name in param_names]
        ax.text(j, i, text, **kwargs_text)
