from functools import wraps
import inspect
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
    def log(self, action=None, feature=None):
        """Logs the function and then runs it

        Parameters
        ----------
        action : string, default=None
            The desired action to be logged (i.e: 'confusion_matrix', 'roc').
            If `action=None` it will log the function's name.

        feature: string, default=None
            The main feature (i.e: 'plot', 'report',
            'SQLiteTracker', 'NotebookCollection')
        """
        def wrapper(func):
            @wraps(func)
            def inner(*args, **kwargs):
                metadata = self._prepare_metadata(
                    self, func, action, feature, *args, **kwargs)
                telemetry.log_api('sklearn-evaluation', metadata=metadata)

                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    metadata['exception'] = str(e)
                    telemetry.log_api(
                        'sklearn-evaluation-error', metadata=metadata)
                    raise e

                return result
            return inner
        return wrapper

    def _get_func_arguments_to_log(self, func, *args, **kwargs):
        args_to_log = dict()
        flags = dict({})

        sig = signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        args_with_default_values = []

        for tupple in sig.parameters.items():
            param_name = tupple[0]
            param_value = tupple[1]
            if param_value.default is not inspect._empty:
                args_with_default_values.append(param_name)

        # extract only args with default values
        for key, value in arguments.items():
            if key in args_with_default_values:
                args_to_log[key] = value

            elif key == 'kwargs':
                flags = self._extract_flags(self, **value)

        return args_to_log, flags

    def _extract_flags(self, **kwargs):
        flags = dict({})
        for key, value in kwargs.items():
            if self._is_flag(self, key):
                flags[key] = value

        return flags

    def _is_flag(self, key):
        flags = self.flags()
        return key in flags

    def _prepare_metadata(self, func, action, feature, *args, **kwargs):
        _action = action or func.__name__
        _args, _flags = self._get_func_arguments_to_log(
            self, func, *args, **kwargs)

        metadata = {
            'action': _action,
            'feature': feature
        }

        if len(_args) > 0:
            metadata['args'] = _args

        if len(_flags) > 0:
            metadata['flags'] = _flags

        return metadata
