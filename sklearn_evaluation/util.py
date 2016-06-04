import re
import collections
from collections import defaultdict, namedtuple
from itertools import product
from six import string_types

import numpy as np


def estimator_type(model):
    s = str(type(model))
    model_name = re.search(".*'(.+?)'.*", s).group(1).split(".")[-1]
    return model_name


def class_name(obj):
    class_name = str(type(obj))
    class_name = re.search(".*'(.+?)'.*", class_name).group(1)
    return class_name


def _can_iterate(obj):
    is_string = isinstance(obj, string_types)
    is_iterable = isinstance(obj, collections.Iterable)

    return is_iterable and not is_string


def is_column_vector(x):
    return len(x.shape) == 2 and x.shape[1] == 1


def is_row_vector(x):
    return len(x.shape) == 1


def _group_by(data, criteria):
    """
        Group objects in data using a function or a key
    """
    if isinstance(criteria, str):
        criteria_str = criteria

        def criteria(x):
            return x[criteria_str]

    res = defaultdict(list)
    for element in data:
        key = criteria(element)
        res[key].append(element)
    return res


def _get_params_value(params):
    """
        Given an iterator (k1, k2), returns a function that when called
        with an object obj returns a tuple of the form:
        ((k1, obj.parameters[k1]), (k2, obj.parameters[k2]))
    """
    # sort params for consistency
    ord_params = sorted(params)

    def fn(obj):
        l = []
        for p in ord_params:
            try:
                l.append((p, obj.parameters[p]))
            except:
                raise ValueError('{} is not a valid parameter'.format(p))
        return tuple(l)
    return fn


def _sorted_map_iter(d):
    ord_keys = sorted(d.keys())
    for k in ord_keys:
        yield (k, d[k])


def _product(k, v):
    """
        Perform the product between two objects
        even if they don't support iteration
    """
    if not _can_iterate(k):
        k = [k]
    if not _can_iterate(v):
        v = [v]
    return list(product(k, v))


def _mapping_to_tuple_pairs(d):
    """
        Convert a mapping object (such as a dictionary) to tuple pairs,
        using its keys and values to generate the pairs and then generating
        all possible combinations between those
        e.g. {1: (1,2,3)} -> (((1, 1),), ((1, 2),), ((1, 3),))
    """
    # order the keys, this will prevent different implementations of Python,
    # return different results from the same dictionary since the order of
    # iteration depends on it
    t = []
    ord_keys = sorted(d.keys())
    for k in ord_keys:
        t.append(_product(k, d[k]))
    return tuple(product(*t))


def _flatten_list(l):
    return [item for sublist in l for item in sublist]


# http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    import numpy as np
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def default_heatmap():
    import matplotlib.pyplot as plt
    return truncate_colormap(plt.cm.OrRd, 0.1, 0.7)


def _dict2named_tuple(d):
    return namedtuple('NamedTupleFromDict', d.keys())(**d)


def _grid_scores_from_dicts(grid_scores):
    # convert every list in cv_validation_scores
    # to a numpy array
    for score in grid_scores:
        val_scores_key = 'cv_validation_scores'
        score[val_scores_key] = np.array(score[val_scores_key])
    # now convert them to named tuples
    grid_scores = [_dict2named_tuple(d) for d in grid_scores]
    return grid_scores
