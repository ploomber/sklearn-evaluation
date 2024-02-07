"""
%load_ext autoreload
%autoreload 2
"""

import matplotlib.pyplot as plt
from sklearn_evaluation.plot.matplotlib.bar import bar_groups
from sklearn_evaluation.plot.matplotlib.data_grid import DataGrid

records = [
    {"a": 1, "b": 10, "c": 1, "data": 0.9},
    {"a": 1, "b": 10, "c": 2, "data": 0.91},
    {"a": 2, "b": 10, "c": 1, "data": 0.95},
    {"a": 2, "b": 10, "c": 2, "data": 0.96},
    {"a": 5, "b": 10, "c": 1, "data": 0.999999},
    {"a": 5, "b": 10, "c": 2, "data": 0.99},
    {"a": 1, "b": 100, "c": 1, "data": 0.7},
    {"a": 1, "b": 100, "c": 2, "data": 0.73},
    {"a": 2, "b": 100, "c": 1, "data": 0.77},
    {"a": 2, "b": 100, "c": 2, "data": 0.76},
    {"a": 5, "b": 100, "c": 1, "data": 0.69},
    {"a": 5, "b": 100, "c": 2, "data": 0.99},
    {"a": 1, "b": 1000, "c": 1, "data": 0.5},
    {"a": 1, "b": 1000, "c": 2, "data": 0.54},
    {"a": 2, "b": 1000, "c": 1, "data": 0.55},
    {"a": 2, "b": 1000, "c": 2, "data": 0.554},
    {"a": 5, "b": 1000, "c": 1, "data": 0.49},
    {"a": 5, "b": 1000, "c": 2, "data": 0.40},
]

bar_groups(
    records,
    group_by=["a", "b"],
    get_value=lambda data: data.mean(),
    get_error=lambda data: data.std(),
)
plt.show()

dg = DataGrid(records, group_by=["a", "b"])
dg.df

list(dg.celliter())

list(dg.rowiter())

# heatmap.heatmap(records)

# import pandas as pd
# df = pd.DataFrame.from_dict(records)

# ax = sns.barplot(x="a", y="data", hue="b", data=df)
# plt.show()
