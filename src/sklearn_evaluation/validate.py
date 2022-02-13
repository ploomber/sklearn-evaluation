from decorator import decorator
from sklearn_evaluation.util import map_parameters_in_fn_call


def argument_is_proportion(argname):
    @decorator
    def argument_is_proportion(func, *args, **kwargs):
        """Validate that an argument is a proportion [0, 1.0]
        """
        arg_maps = map_parameters_in_fn_call(args, kwargs, func)
        value = arg_maps.get(argname)

        # Validate value, but only if has a value
        if (not (0 <= value <= 1.0)) and value is not None:
            raise ValueError(
                '{argname} must be between 0 and 1.0'.format(argname=argname))
        return func(*args, **kwargs)

    return argument_is_proportion


@decorator
def proportion(func, *args, **kwargs):
    # Get proportion parameter, look in kwargs and args
    arg_map = map_parameters_in_fn_call(args, kwargs, func)
    proportion = arg_map.get('proportion')

    # Validate proportion, but only if has a value
    if (not (0 <= proportion <= 1.0)) and proportion is not None:
        raise ValueError('Proportion must be between 0 and 1.0')
    return func(*args, **kwargs)
