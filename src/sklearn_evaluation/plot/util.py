from decorator import decorator

import matplotlib.pyplot as plt

from sklearn_evaluation import util


@decorator
def set_default_ax(func, *args, **kwargs):
    params = util.map_parameters_in_fn_call(args, kwargs, func)

    if 'ax' not in params:
        raise Exception('ax is not a parameter in {}'.format(func))

    if params['ax'] is None:
        params['ax'] = plt.gca()

    return func(**params)


def requires_properties(properties):

    @decorator
    def _requires_properties(func, *args, **kwargs):
        params = util.map_parameters_in_fn_call(args, kwargs, func)
        obj = params.get('self')

        if obj is None:
            raise Exception('This decorator only works on instance methods')

        missing = [p for p in properties if getattr(obj, p) is None]

        if len(missing):
            raise ValueError('{} requires {} to be set, missing: {}'
                             .format(func.__name__, properties, missing))

        return func(*args, **kwargs)

    return _requires_properties
