"""
"""
import matplotlib.pyplot as plt
from sklearn_evaluation.plot.matplotlib import heatmap

records = [
    {'a': 1, 'b': 10, 'data': 0.9},
    {'a': 2, 'b': 10, 'data': 0.95},
    {'a': 5, 'b': 10, 'data': 0.999999},

    {'a': 1, 'b': 100, 'data': 0.7},
    {'a': 2, 'b': 100, 'data': 0.77},
    {'a': 5, 'b': 100, 'data': 0.69},

    {'a': 1, 'b': 1000, 'data': 0.5},
    {'a': 2, 'b': 1000, 'data': 0.55},
    {'a': 5, 'b': 1000, 'data': 0.49},
]

heatmap.heatmap(records)
plt.show()
