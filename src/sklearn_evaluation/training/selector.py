"""
When training models, it is common to try out different
subsets of features or subpopulations. ``DataSelector`` allows you to define
a series of transformations on your data so you can succinctly define a
subsetting pipeline as a series of dictionaries.
"""
from copy import copy, deepcopy
import abc
import inspect
import importlib
import itertools
from collections.abc import Mapping

import pandas as pd
from decorator import decorator

from sklearn_evaluation.exceptions import DataSelectorError
from sklearn_evaluation.util import map_parameters_in_fn_call
from sklearn_evaluation.table import Table


def import_from_dotted_path(dotted_path):
    parts = dotted_path.split('.')
    mod_name, callable_ = '.'.join(parts[:-1]), parts[-1]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, callable_)
    return fn


def expand_value(value):
    """
    If value is a str with at least one dot ("."), try to import it and call
    it, if anything fails, return the value
    """
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


# NOTE: consider deleting
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


def concatenate_over(argname):
    """Decorator to "vectorize" functions and concatenate outputs
    """
    def _prepare_args(arg_map, value):
        params = copy(arg_map)
        params[argname] = value
        return params

    @decorator
    def _concatenate_over(func, *args, **kwargs):
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

    return _concatenate_over


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


@concatenate_over('prefix')
def _with_prefix(df, prefix):
    return [] if not prefix else [
        c for c in df.columns if c.startswith(prefix)
    ]


@concatenate_over('suffix')
def _with_suffix(df, suffix):
    return [] if not suffix else [c for c in df.columns if c.endswith(suffix)]


@concatenate_over('substr')
def _contains(df, substr):
    return [] if not substr else [c for c in df.columns if substr in c]


def _with_max_na_prop(df, max_prop):
    if max_prop is not None:
        na_prop = df.isna().sum(axis='index') / len(df)
        return na_prop[na_prop > max_prop].index.tolist()
    else:
        return []


class ColumnDrop(Step):
    """Drop columns

    Parameters
    ----------
    names
        List of columns to drop
    prefix
        Drop columns with this prefix (or list of)
    suffix
        Drop columns with this suffix (or list of)
    contains
        Drop columns if they contains this substring
    max_na_prop
        Drop columns whose proportion of NAs [0, 1] is larger than this

    """
    @expand_arguments
    def __init__(self,
                 names: list = None,
                 prefix: str = None,
                 suffix: str = None,
                 contains: str = None,
                 max_na_prop: float = None):
        self.names = names or []
        self.prefix = prefix
        self.suffix = suffix
        self.contains = contains
        self.max_na_prop = max_na_prop
        self.to_delete_ = None

    def transform(self, df, return_summary=False):
        self.to_delete_ = set(self.names + _with_prefix(df, self.prefix) +
                              _with_suffix(df, self.suffix) +
                              _with_max_na_prop(df, self.max_na_prop) +
                              _contains(df, self.contains))

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
    """Drop rows

    Parameters
    ----------
    if_nas
        If True, deletes all rows where there is at leat one NA
    query
        Drops all rows matching the query (passed via pandas.query)
    """
    @expand_arguments
    def __init__(self, if_nas: bool = False, query: str = None):
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
    """Subset columns

    Parameters
    ----------
    names
        List of columns to keep
    """
    def __init__(self, names: list = None, dotted_path: str = None):
        self.names = names or []
        self.dotted_path = dotted_path

    def transform(self, df, return_summary=False):
        to_keep = copy(self.names)

        if self.dotted_path:
            fn = import_from_dotted_path(self.dotted_path)
            to_keep.extend(fn(df))
            # remove duplicates
            to_keep = list(set(to_keep))

        return df[to_keep], self.transform_summary(to_keep)

    def transform_summary(self, to_keep):
        return 'Keeping {:,} column(s)'.format(len(to_keep))


class DataSelector:
    """Subset a pandas.DataFrame by passing a series of steps

    Parameters
    ----------
    *steps
        Steps to apply to the data sequentially (order matters). Each step
        must be a dictionary with a key "kind" whose value must be one of
        "column_drop", "row_drop" or "column_keep". The rest of the key-value
        pairs must match the signature for the corresponding Step objects

    """
    def __init__(self, *steps):
        steps = deepcopy(steps)
        self.steps = [_instantiate_step(step) for step in steps]

    def transform(self, df, return_summary: bool = False):
        """Apply steps

        Parameters
        ----------
        df
            Data frame to transform
        return_summary
            If False, the function only returns the output data frame,
            if True, it also returns a summary table
        """
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
            table = Table([(type(step).__name__, summary)
                           for step, summary in zip(self.steps, summaries)],
                          header=['Step', 'Summary'])
            return result, table

    def _get_table(self):
        return Table([(type(step).__name__, step.get_args(), step.get_params())
                      for step in self.steps],
                     header=['Step', 'Args', 'Params'])

    def __repr__(self):
        table = str(self._get_table())
        table = '{} with steps:\n'.format(type(self).__name__) + table
        return table

    def _repr_html_(self):
        return self._get_table().to_html()


def _instantiate_step(step):
    if not isinstance(step, Mapping):
        raise TypeError('step must be a mapping, got {}'.format(
            type(step).__name__))

    kind = step.pop('kind', None)

    if kind not in _mapping:
        raise ValueError('Each step must have a kind key with one of '
                         'the valid values: {}'.format(set(_mapping)))

    return _mapping[kind](**step)


_mapping = {
    'column_drop': ColumnDrop,
    'row_drop': RowDrop,
    'column_keep': ColumnKeep,
}
