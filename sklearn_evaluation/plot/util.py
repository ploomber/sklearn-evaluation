from decorator import decorator

import matplotlib.pyplot as plt

from sklearn_evaluation import util


@decorator
def set_default_ax(func, *args, **kwargs):
    params = util.map_parameters_in_fn_call(args, kwargs, func)

    if 'ax' not in params:
        raise Exception(f'ax is not a parameter in {func}')

    if params['ax'] is None:
        params['ax'] = plt.gca()

    return func(**params)
