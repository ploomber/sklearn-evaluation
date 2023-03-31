import pytest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from matplotlib.testing.decorators import image_comparison

from sklearn_evaluation.plot.pca import pca

X, y = make_classification(
    n_samples=5000, n_features=8, n_informative=6, class_sep=3.0, random_state=0
)


@image_comparison(
    baseline_images=["pca_custom_colors"], extensions=["png"], remove_text=False
)
def test_pca_custom_colors():
    pca(X, y, colors=["r", "b"])


@image_comparison(baseline_images=["pca"], extensions=["png"], remove_text=False)
def test_pca():
    pca(X, y)


@image_comparison(
    baseline_images=["pca_without_y"], extensions=["png"], remove_text=False
)
def test_pca_without_y():
    pca(X)


@image_comparison(
    baseline_images=["pca_target_names"], extensions=["png"], remove_text=False
)
def test_pca_target_names():
    pca(X, y, target_names=["Class A", "Class B"])


@image_comparison(
    baseline_images=["n_components_one", "n_components_two", "n_components_three"],
    extensions=["png"],
    remove_text=False,
)
def test_ncomponents():
    pca(X, y, target_names=["Class A", "Class B"], n_components=3)


def test_pca_random_colors():
    pca(X, y)


@image_comparison(
    baseline_images=["pca_multiclass"], extensions=["png"], remove_text=False
)
def test_pca_multiclass():
    X, y = make_classification(
        n_samples=5000,
        n_classes=3,
        n_features=8,
        n_informative=6,
        class_sep=3.0,
        random_state=0,
    )
    pca(X, y, target_names=["Class A", "Class B", "Class C"])


def test_ax_pca():
    fig, ax = plt.subplots(1, 1)
    out_ax = pca(X, y)
    assert ax is not out_ax
    out_ax = pca(X, y, ax=[ax])
    assert ax is out_ax[0]


def test_nan_error(ploomber_value_error_message):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        pca(np.array([[1, float("nan"), 2], [3, 0, 6]]), np.array([1, 2]))


def test_component_error(ploomber_value_error_message):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        pca(X, y, n_components=1)


def test_target_color_mismatch(ploomber_value_error_message):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        pca(X, y, target_names=["A", "B"], colors=["r"])


def test_empty_input_error(ploomber_value_error_message):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        pca([], [])
