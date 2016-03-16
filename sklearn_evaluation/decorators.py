import inspect
import functools

def validate_proportion(func):
    @functools.wraps(func)
    def validated_func(*args, **kwargs):
        #Get proportion parameter, look in kwargs and args
        try:
            proportion = kwargs['proportion']
        except:
            fn_args, _, _, _ = inspect.getargspec(func)
            idx = fn_args.index('proportion')
            proportion = args[idx]
        #Validate proportion
        if not (0 <= proportion <= 1.0):
            raise ValueError('Proportion must be between 0 and 1.0')
        return func(*args, **kwargs)
    return validated_func