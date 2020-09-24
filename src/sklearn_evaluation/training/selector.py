import abc
import inspect
import importlib

import pandas as pd
from tabulate import tabulate
from decorator import decorator


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


class Step(abc.ABC):
    @abc.abstractclassmethod
    def fit(self, df):
        pass

    @abc.abstractmethod
    def transform(self, df):
        pass

    def get_args(self):
        args = inspect.getfullargspec(self.__init__).args
        args.remove('self')
        return {arg: getattr(self, arg) for arg in args}

    def get_params(self):
        return {k: v for k, v in self.__dict__.items() if k.endswith('_')}


def _with_prefix(df, prefix):
    return [] if not prefix else [
        c for c in df.columns if c.startswith(prefix)
    ]


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

    def fit(self, df):
        self.to_delete_ = set(self.names + _with_prefix(df, self.prefix) +
                              _with_suffix(df, self.suffix) +
                              _with_max_na_prop(df, self.max_na_prop))
        return self

    def transform(self, df, return_summary=False):
        out = df[[c for c in df.columns if c not in self.to_delete_]]
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

    def fit(self, df):
        return self

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


class RowKeep(Step):
    @expand_arguments
    def __init__(self, keep):
        self.keep = keep

    def fit(self, df):
        return self

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

        for step in self.steps:
            if return_summary:
                result, summary = step.transform(result, return_summary=True)
                summaries.append(summary)
            else:
                result = step.transform(result, return_summary=False)

        if not return_summary:
            return result
        else:
            table = tabulate([(type(step).__name__, summary)
                              for step, summary in zip(self.steps, summaries)],
                             headers=['Step', 'Summary'],
                             tablefmt='grid')
            return result, table

    def fit(self, df):
        result = df

        for step in self.steps:
            result = step.fit(result)

        return self

    def fit_transform(self, df, return_summary=False):
        return self.fit(df).transform(df, return_summary=return_summary)

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
    'row_keep': RowKeep,
}
