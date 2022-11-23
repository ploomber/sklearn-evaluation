import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from sklearn_evaluation.plot import rank_one_dimensional, rank_one_dimensional_from_results

iris = load_iris()
features = iris['feature_names']
X = pd.DataFrame(data=np.c_[iris['data']], columns=features)


def test_unknown_algorithm_1d():
    with pytest.raises(ValueError) as e:
        rank_one_dimensional(X, features, algorithm='unknown')
        assert "'unknown' is unrecognized ranking method" in str(e.value)


@image_comparison(baseline_images=['shapiro_1d_iris_horizontal'],
                  extensions=['png'],
                  remove_text=False)
def test_1d_horizontal():
    rank_one_dimensional(X,
                         features,
                         algorithm='shapiro',
                         orientation='h',
                         figsize=(13, 7))


@image_comparison(baseline_images=['shapiro_1d_iris_vertical'],
                  extensions=['png'],
                  remove_text=False)
def test_1d_vertical():
    rank_one_dimensional(X,
                         features,
                         algorithm='shapiro',
                         orientation='v',
                         figsize=(7, 15))


@image_comparison(baseline_images=['shapiro_1d_iris_numpy'],
                  extensions=['png'],
                  remove_text=False)
def test_1d_numpy():
    X_numpy = X.to_numpy()
    assert isinstance(X_numpy, np.ndarray)
    rank_one_dimensional(X_numpy,
                         algorithm='shapiro',
                         orientation='h',
                         figsize=(13, 7))


@image_comparison(baseline_images=['shapiro_1d_iris_numpy_features'],
                  extensions=['png'],
                  remove_text=False)
def test_1d_numpy_features():
    X_numpy = X.to_numpy()
    assert isinstance(X_numpy, np.ndarray)
    rank_one_dimensional(X_numpy,
                         features=features,
                         algorithm='shapiro',
                         orientation='h',
                         figsize=(13, 7))


@image_comparison(baseline_images=['shapiro_1d_iris_horizontal_color'],
                  extensions=['png'],
                  remove_text=False)
def test_1d_horizontal_color():
    rank_one_dimensional(X,
                         features,
                         algorithm='shapiro',
                         color='b',
                         orientation='h',
                         figsize=(13, 7))
    plt.savefig('shapiro_1d_iris_horizontal_color.png')
