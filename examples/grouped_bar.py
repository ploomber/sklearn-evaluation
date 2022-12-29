"""
"""
import matplotlib.pyplot as plt
from sklearn_evaluation.plot.matplotlib.bar import bar_groups

records = [
    {"a": 1, "b": 10, "data": 0.9},
    {"a": 2, "b": 10, "data": 0.95},
    {"a": 5, "b": 10, "data": 0.99},
    {"a": 1, "b": 100, "data": 0.4},
    {"a": 2, "b": 100, "data": 0.44},
    {"a": 5, "b": 100, "data": 0.41},
    {"a": 1, "b": 1000, "data": 0.1},
    {"a": 2, "b": 1000, "data": 0.11},
    {"a": 5, "b": 1000, "data": 0.09},
]


ax = plt.gca()
bar_groups(records, ax)
plt.show()
