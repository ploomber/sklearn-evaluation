import numpy as np
import matplotlib.pyplot as plt
import os
from src.sklearn_evaluation.plot.plot import AbstractPlot


class ConcretePlot(AbstractPlot):
    def __init__(self, *, label=None):
        pass

    def plot(self, ax=None):
        pass

    def _get_data(self):
        pass

    def from_raw_data(cls):
        pass


def test_to_png():
    x = np.linspace(0, 30, 100)
    y = np.cos(x)
    plt.plot(x, y)
    new_plot = ConcretePlot()
    new_plot.figure_ = plt.gcf()

    path_name = "test_plot.png"
    new_plot.to_png(path_name)

    assert os.path.exists(path_name)
    assert os.path.isfile(path_name)
    assert os.path.getsize(path_name) > 0
    os.remove(path_name)
