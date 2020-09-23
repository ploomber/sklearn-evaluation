import abc
import inspect

import pandas as pd
from tabulate import tabulate


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

    def transform(self, df):
        return df[[c for c in df.columns if c not in self.to_delete_]]


def _incomplete_cases(df):
    nas = df.isna().sum(axis='columns')
    return nas[nas > 0].index


def _query(df, query):
    return df.query(query).index


class RowDrop(Step):
    def __init__(self, if_nas=False, query=None):
        self.if_nas = if_nas
        self.query = query

    def fit(self, df):
        return self

    def transform(self, df):
        to_delete_ = pd.Index([])

        if self.if_nas:
            to_delete_ = to_delete_.union(_incomplete_cases(df))

        if self.query:
            to_delete_ = to_delete_.union(_query(df, self.query))

        return df[~df.index.isin(to_delete_)]


class DataSelector:
    def __init__(self, steps=None):
        steps = steps or []
        self.steps = [_mapping[step[0]](**step[1]) for step in steps]

    def transform(self, df):
        result = df

        for step in self.steps:
            result = step.transform(result)

        return result

    def fit(self, df):
        result = df

        for step in self.steps:
            result = step.fit(result)

        return self

    def fit_transform(self, df):
        return self.fit(df).transform(df)

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
}
