import re
import types
import collections


def estimator_type(model):
    s = str(type(model))
    model_name = re.search(".*'(.+?)'.*", s).group(1).split(".")[-1]
    return model_name


def class_name(obj):
    class_name = str(type(obj))
    class_name = re.search(".*'(.+?)'.*", class_name).group(1)
    return class_name


def _can_iterate(obj):
    is_string = isinstance(obj, types.StringTypes)
    is_iterable = isinstance(obj, collections.Iterable)

    return is_iterable and not is_string
