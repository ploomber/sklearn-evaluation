import inspect
from decorator import decorator


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
