from functools import wraps
from inspect import signature

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from importlib_metadata import version

from ploomber_core.telemetry.telemetry import Telemetry

telemetry = Telemetry(
    api_key="phc_P9SpSeypyPwxrMdFn2edOOEooQioF2axppyEeDwtMSP",
    package_name="sklearn-evaluation",
    version=version('sklearn_evaluation'),
)


class SKLearnEvaluationLogger():
    def flags():
        return ['is_report']

    @classmethod
    def log(self, func=None, action=None, feature=None):
        if callable(func):
            @wraps(func)
            def inner(*args, **kwargs):
                self._log_api(self, func, action, feature, *args, **kwargs)
                return func(*args, **kwargs)

            return inner
        else:
            def wrapper(func):
                @wraps(func)
                def inner(*args, **kwargs):
                    self._log_api(self, func, action, feature, *args, **kwargs)
                    return func(*args, **kwargs)
                return inner
            return wrapper

    def _get_func_arguments(func, *args, **kwargs):
        sig = signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments

        return arguments

    def _parse_arguments(self, arguments):
        parsed_function_arguments = dict()
        flags = dict()

        kwargs = arguments.pop('kwargs', {})

        for key, value in arguments.items():
            parsed_function_arguments[key] = str(value)

        for key, value in kwargs.items():
            if self._is_flag(self, key):
                flags[key] = value
            else:
                parsed_function_arguments[key] = str(value)

        return dict({
            'function_args': parsed_function_arguments,
            'flags': flags
        })

    def _log_api(self, func, action, feature, *args, **kwargs):
        _action = action or func.__name__
        arguments = self._get_func_arguments(func, *args, **kwargs)
        _arguments = self._parse_arguments(self, arguments)

        metadata = {
            'action': _action,
            'feature': feature,
            'function_arguments': _arguments['function_args'],
        }

        if len(_arguments['flags']) > 0:
            metadata['flags'] = _arguments['flags']

        telemetry.log_api('sklearn-evaluation', metadata=metadata)

    def _is_flag(self, key):
        flags = self.flags()
        return key in flags
