import inspect
from decorator import decorator


def argument_is_proportion(argname):

    @decorator
    def argument_is_proportion(func, *args, **kwargs):
        """Validate that an agument is a proportion [0, 1.0]
        """

        try:
            value = kwargs[argname]
        except Exception:
            try:
                fn_args = inspect.getargspec(func).args
                idx = fn_args.index(argname)
                value = args[idx]
            except Exception:
                value = None
        # Validate value, but only if has a value
        if (not (0 <= value <= 1.0)) and value is not None:
            raise ValueError('{argname} must be between 0 and 1.0'
                             .format(argname=argname))
        return func(*args, **kwargs)

    return argument_is_proportion


@decorator
def proportion(func, *args, **kwargs):
    # Get proportion parameter, look in kwargs and args
    try:
        proportion = kwargs['proportion']
    except:
        try:
            fn_args = inspect.getargspec(func).args
            idx = fn_args.index('proportion')
            proportion = args[idx]
        except:
            proportion = None
    # Validate proportion, but only if has a value
    if (not (0 <= proportion <= 1.0)) and proportion is not None:
        raise ValueError('Proportion must be between 0 and 1.0')
    return func(*args, **kwargs)
