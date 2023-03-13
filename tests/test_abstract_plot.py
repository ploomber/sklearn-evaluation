"""
Tests for AbstractPlot class
"""
import pytest

from sklearn_evaluation.plot.plot import AbstractPlot


class testPlot(AbstractPlot):
    def __init__(self, *, label=None):
        pass

    def plot(self, ax=None):
        pass

    def from_raw_data(cls):
        pass

    def _get_data(self):
        pass


def test_add():
    test1 = testPlot()
    test2 = testPlot()
    with pytest.raises(NotImplementedError) as excinfo:
        def f():
            test1 + test2
        f()
    assert (f"{type(test1).__name__!r} doesn't support the add (+) operator"
            in str(excinfo.value))


def test_sub():
    test1 = testPlot()
    test2 = testPlot()
    with pytest.raises(NotImplementedError) as excinfo:
        def f():
            test1 - test2
        f()
    assert (f"{type(test1).__name__!r} doesn't support the subtract (-) operator"
            in str(excinfo.value))
