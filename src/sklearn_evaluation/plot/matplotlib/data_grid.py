import numpy as np
import pandas as pd


class DataGrid:

    def __init__(self, records):
        # build data frame
        df = pd.DataFrame.from_dict(records)

        # columns that will be interpreted as paramers
        params = sorted(set(df.columns) - set(['data']))

        if len(params) != 2:
            raise ValueError('There should be exactly two columns apart from '
                             f'data, found: {len(params)} ({params})')

        # fill with nas if any combination is missing
        prod = pd.MultiIndex.from_product([df[params[0]].unique(),
                                           df[params[1]].unique()],
                                          names=params)

        df = (df.set_index(params)
              .reindex(prod, fill_value=np.nan).reset_index())

        # sort by param colums
        self.shape = len(df[params[0]].unique()), len(df[params[1]].unique())
        self.df = df.sort_values(params)
        self.params = params

    def celliter(self):
        # supply data by grouping on the first parameter
        for i, (name, values) in enumerate(self.rowiter()):
            for j, value in enumerate(values):
                yield i, j, value

    def rowiter(self):
        # supply data by grouping on the first parameter
        for name, group in self.df.groupby(self.params[0]):
            yield name, group['data'].values

    def rownames(self):
        values = self.df[self.params[0]].unique()
        values.sort()
        return values

    def colnames(self):
        values = self.df[self.params[1]].unique()
        values.sort()
        return values
