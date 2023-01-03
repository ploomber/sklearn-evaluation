import matplotlib.pyplot as plt
import pytest
import numpy as np
import pandas as pd
from matplotlib.testing.decorators import image_comparison
from sklearn.datasets import make_regression

from sklearn_evaluation import plot

y_true = np.array([105, 120, 120, 160, 120, 145, 175, 160, 185, 210, 150])
y_pred = np.array(
    [
        108.19,
        115.16,
        122.13,
        136.06,
        136.06,
        156.97,
        163.94,
        170.91,
        184.84,
        205.75,
        151.23,
    ]
)

X, y = make_regression(
    n_samples=100,
    n_features=14,
    n_informative=6,
    bias=1.2,
    noise=49.8,
    tail_strength=0.6,
    random_state=637,
)


@image_comparison(baseline_images=["residual"], extensions=["png"], remove_text=True)
def test_residuals():
    plot.residuals(y_true, y_pred)


@pytest.mark.parametrize(
    "y_true, y_pred",
    [(y_true, y_pred), (pd.Series(y_true), pd.Series(y_pred))],
    ids=["numpy", "pandas"],
)
@image_comparison(
    baseline_images=["prediction_error"], extensions=["png"], remove_text=True
)
def test_prediction_error(y_true, y_pred):
    plot.prediction_error(y_true, y_pred)


@image_comparison(
    baseline_images=["prediction_error_lines_one"],
    extensions=["png"],
    remove_text=True,
    tol=1.2,
)
def test_prediction_error_lines_one(regression_data_one):
    y_true, y_pred = regression_data_one
    plot.prediction_error(y_true, y_pred)


@image_comparison(
    baseline_images=["prediction_error_lines_two"],
    extensions=["png"],
    remove_text=True,
    tol=0.1,
)
def test_prediction_error_lines_two(regression_data_two):
    y_true, y_pred = regression_data_two
    plot.prediction_error(y_true, y_pred)


@image_comparison(
    baseline_images=["cooks_distance"], extensions=["png"], remove_text=True
)
def test_cooks_distance():
    plot.cooks_distance(X, y)


def test_cooks_distance_ax():
    out_ax = plot.cooks_distance(X, y)
    fig, ax = plt.subplots(1, 1)
    assert out_ax is not ax
    out_ax = plot.cooks_distance(X, y, ax=ax)
    assert ax is out_ax
