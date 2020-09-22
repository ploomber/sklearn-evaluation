import numpy as np
import pandas as pd


class DataGrid:
    def __init__(self, records, group_by=None):
        # build data frame
        df = pd.DataFrame.from_dict(records)

        # columns that will be interpreted as paramers
        params = sorted(set(df.columns) - set(['data']))

        if group_by is None:
            if len(params) != 2:
                raise ValueError('There should be exactly two columns apart '
                                 f'from data, found: {len(params)} ({params}) '
                                 'if group_by is not specified')
            else:
                group_by = params
        else:
            if len(group_by) != 2:
                raise ValueError('group_by must have 2 elements, '
                                 f'got: {len(group_by)}')

        self.params = params
        self.group_by = group_by

        # fill with nas if any combination is missing
        unique = [df[p].unique().tolist() for p in params]
        prod = pd.MultiIndex.from_product(unique, names=params)

        df = (df.set_index(params).reindex(prod,
                                           fill_value=np.nan).reset_index())

        self.shape = len(df[group_by[0]].unique()), len(
            df[group_by[1]].unique())
        self.df = df.sort_values(group_by)

    def celliter(self):
        # supply data by grouping on the first parameter
        for i, (name, values) in enumerate(self.rowiter()):
            for j, value in enumerate(values):
                yield i, j, value

    def rowiter(self):
        # supply data by grouping on the first parameter
        for name, group in self.df.groupby(self.group_by[0]):
            # if there are only three columns (param1, param2, data),
            # then ['data'].values will return the values for a fixed param1
            # in increasing param2 order
            if len(self.params) == 2:
                yield name, group['data'].values
            # if there are more than one params, then we have to do a second
            # group_by, so the values are grouped together for a fixed
            # (param1, param2) combination
            else:
                values = [
                    df_sub['data'].values
                    for _, df_sub in group.groupby(self.group_by[1])
                ]
                yield name, values

    def rownames(self):
        values = self.df[self.group_by[0]].unique()
        values.sort()
        return values

    def colnames(self):
        values = self.df[self.group_by[1]].unique()
        values.sort()
        return values
