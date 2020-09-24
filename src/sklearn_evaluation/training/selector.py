from copy import copy
import abc
import inspect
import importlib
import itertools

import pandas as pd
from tabulate import tabulate
from decorator import decorator

from sklearn_evaluation.exceptions import DataSelectorError
from sklearn_evaluation.util import map_parameters_in_fn_call


def expand_value(value):
    if isinstance(value, str) and '.' in value:
        parts = value.split('.')
        mod_name, callable = '.'.join(parts[:-1]), parts[-1]

        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            return value

        try:
            fn = getattr(mod, callable)
        except AttributeError:
            return value

        return fn()
    else:
        return value


@decorator
def expand_arguments(func, *args, **kwargs):
    """
    Fnctions decorated with expand_argument call "expand_value" on each
    passed argument, which will interpred as a "dotted path" any string with
    dots on it and replace the value by the value returned by a function
    imported from that location (no arguments passed), if no function
    is found in such location, the original value is returned
    """
    return func(*[expand_value(arg) for arg in args],
                **{k: expand_value(v)
                   for k, v in kwargs.items()})


def union_over(argname):
    def _prepare_args(arg_map, value):
        params = copy(arg_map)
        params[argname] = value
        return params

    @decorator
    def _union_over(func, *args, **kwargs):
        """Validate that an agument is a proportion [0, 1.0]
        """
        arg_map = map_parameters_in_fn_call(args, kwargs, func)
        value = arg_map.get(argname)

        if isinstance(value, list):
            return list(
                itertools.chain.from_iterable(
                    func(**_prepare_args(arg_map, v)) for v in value))
        else:
            return func(**arg_map)

    return _union_over


class Step(abc.ABC):
    @abc.abstractmethod
    def transform(self, df):
        pass

    def get_args(self):
        args = inspect.getfullargspec(self.__init__).args
        args.remove('self')
        return {arg: getattr(self, arg) for arg in args}

    # NOTE: consider deleting this, just show if in the transform summary
    def get_params(self):
        return {k: v for k, v in self.__dict__.items() if k.endswith('_')}


@union_over('prefix')
def _with_prefix(df, prefix):
    return [] if not prefix else [
        c for c in df.columns if c.startswith(prefix)
    ]


@union_over('suffix')
def _with_suffix(df, suffix):
    return [] if not suffix else [c for c in df.columns if c.endswith(suffix)]


def _with_max_na_prop(df, max_prop):
    if max_prop is not None:
        na_prop = df.isna().sum(axis='index') / len(df)
        return na_prop[na_prop > max_prop].index.tolist()
    else:
        return []


class ColumnDrop(Step):
    @expand_arguments
    def __init__(self, names=None, prefix=None, suffix=None, max_na_prop=None):
        self.names = names or []
        self.prefix = prefix
        self.suffix = suffix
        self.max_na_prop = max_na_prop
        self.to_delete_ = None

    def transform(self, df, return_summary=False):
        self.to_delete_ = set(self.names + _with_prefix(df, self.prefix) +
                              _with_suffix(df, self.suffix) +
                              _with_max_na_prop(df, self.max_na_prop))

        out = df.drop(self.to_delete_, axis='columns')
        return out if not return_summary else (out, self.transform_summary(df))

    def transform_summary(self, df):
        return 'Deleted {:,} columns: {}'.format(len(self.to_delete_),
                                                 self.to_delete_)


def _incomplete_cases(df):
    nas = df.isna().sum(axis='columns')
    return nas[nas > 0].index


def _query(df, query):
    return df.query(query).index


class RowDrop(Step):
    @expand_arguments
    def __init__(self, if_nas=False, query=None):
        self.if_nas = if_nas
        self.query = query

    def transform(self, df, return_summary=False):
        to_delete = pd.Index([])

        if self.if_nas:
            to_delete = to_delete.union(_incomplete_cases(df))

        if self.query:
            to_delete = to_delete.union(_query(df, self.query))

        out = df[~df.index.isin(to_delete)]

        return out if not return_summary else (out,
                                               self.transform_summary(
                                                   df, to_delete))

    def transform_summary(self, df, to_delete):
        n = len(to_delete)
        return 'Deleted {:,} rows ({:.1%})'.format(n, n / len(df))


class ColumnKeep(Step):
    @expand_arguments
    def __init__(self, keep):
        self.keep = keep

    def transform(self, df, return_summary=False):
        return df[self.keep], self.transform_summary()

    def transform_summary(self):
        return 'Keeping {:,} column(s)'.format(len(self.keep))


class DataSelector:
    def __init__(self, steps=None):
        steps = steps or []
        self.steps = [_mapping[step[0]](**step[1]) for step in steps]

    def transform(self, df, return_summary=False):
        result = df
        summaries = []

        for i, step in enumerate(self.steps):
            try:
                result = step.transform(result, return_summary=return_summary)
            except Exception as e:
                raise DataSelectorError('Error executing step {} ({})'.format(
                    i,
                    type(step).__name__)) from e

            if return_summary:
                result, summary = result
                summaries.append(summary)

        if not return_summary:
            return result
        else:
            table = tabulate([(type(step).__name__, summary)
                              for step, summary in zip(self.steps, summaries)],
                             headers=['Step', 'Summary'],
                             tablefmt='grid')
            return result, table

    def __repr__(self):
        table = tabulate(
            [(type(step).__name__, step.get_args(), step.get_params())
             for step in self.steps],
            headers=['Step', 'Args', 'Params'],
            tablefmt='grid')
        table = '{} with steps:\n'.format(type(self).__name__) + table
        return table


_mapping = {
    'column_drop': ColumnDrop,
    'row_drop': RowDrop,
    'column_keep': ColumnKeep,
}
