"""
"""
import matplotlib.pyplot as plt
from sklearn_evaluation.plot.matplotlib import heatmap

sample_data = [
    {'params': {'a': 1, 'b': 10}, 'value': 0.9},
    {'params': {'a': 2, 'b': 10}, 'value': 0.95},
    {'params': {'a': 5, 'b': 10}, 'value': 0.99},

    {'params': {'a': 1, 'b': 100}, 'value': 0.7},
    {'params': {'a': 2, 'b': 100}, 'value': 0.77},
    {'params': {'a': 5, 'b': 100}, 'value': 0.69},

    {'params': {'a': 1, 'b': 1000}, 'value': 0.5},
    {'params': {'a': 2, 'b': 1000}, 'value': 0.55},
    {'params': {'a': 5, 'b': 1000}, 'value': 0.49},
]


heatmap.heatmap(sample_data)
plt.show()
