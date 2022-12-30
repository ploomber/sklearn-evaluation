"""
NOTE: Some of the test cases are based on the yellowbricks test module.
License below.

Copyright 2016-2020 The scikit-yb developers
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import matplotlib.pyplot as plt
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from matplotlib.testing.decorators import image_comparison

from sklearn_evaluation.plot import Rank1D, Rank2D

iris = load_iris()
features = iris["feature_names"]
X = pd.DataFrame(data=np.c_[iris["data"]], columns=features)
rank1d = Rank1D()


def test_unknown_algorithm_1d():
    with pytest.raises(ValueError) as e:
        Rank1D(algorithm="unknown").feature_ranks(X)
        assert "'unknown' is unrecognized ranking method" in str(e.value)


@image_comparison(
    baseline_images=["shapiro_1d_iris_horizontal"],
    extensions=["png"],
    remove_text=False,
)
def test_1d_horizontal():
    Rank1D(
        algorithm="shapiro", features=features, orient="h", figsize=(13, 7)
    ).feature_ranks(X)


@image_comparison(
    baseline_images=["shapiro_1d_iris_vertical"], extensions=["png"], remove_text=False
)
def test_1d_vertical():
    Rank1D(
        algorithm="shapiro", features=features, orient="v", figsize=(7, 15)
    ).feature_ranks(X)


def test_1d_invalid_orientation():
    with pytest.raises(ValueError) as e:
        Rank1D(
            algorithm="shapiro",
            features=features,
            orient="invalid",
        ).feature_ranks(X)
        assert "Orientation must be 'h' or 'v'" in str(e.value)


def test_1d_incorrect_features():
    with pytest.raises(ValueError) as e:

        Rank1D(
            features=["Sepal length", "Sepal width"], algorithm="shapiro"
        ).feature_ranks(X)
        assert (
            "number of supplied feature names does not match the number "
            "of columns in the training data." in str(e.value)
        )


@image_comparison(
    baseline_images=["shapiro_1d_iris_numpy"], extensions=["png"], remove_text=False
)
def test_1d_numpy():
    X_numpy = X.to_numpy()
    assert isinstance(X_numpy, np.ndarray)
    Rank1D(algorithm="SHAPIRO", orient="h", figsize=(13, 7)).feature_ranks(X_numpy)


@image_comparison(
    baseline_images=["shapiro_1d_iris_numpy_features"],
    extensions=["png"],
    remove_text=False,
)
def test_1d_numpy_features():
    X_numpy = X.to_numpy()
    assert isinstance(X_numpy, np.ndarray)
    Rank1D(
        features=features, algorithm="shapiro", orient="h", figsize=(13, 7)
    ).feature_ranks(X_numpy)


@image_comparison(
    baseline_images=["shapiro_1d_iris_horizontal_color"],
    extensions=["png"],
    remove_text=False,
)
def test_1d_horizontal_color():
    Rank1D(
        features=features, algorithm="shapiro", color="b", orient="h", figsize=(13, 7)
    ).feature_ranks(X)


@image_comparison(
    baseline_images=["shapiro_from_results"], extensions=["png"], remove_text=False
)
def test_1d_custom_ranks():
    ranks = np.array([0.96784323, 0.79394983, 0.47927427])
    Rank1D(
        features=["Feature1", "Feature2", "Feature3"], orient="h", figsize=(13, 7)
    ).feature_ranks_custom_algorithm(ranks)


def test_1d_custom_ranks_incorrect_features():
    ranks = np.array([0.96784323, 0.79394983, 0.47927427])
    with pytest.raises(ValueError) as e:
        Rank1D(
            features=["Feature1", "Feature2", "Feature3", "Feature4"]
        ).feature_ranks_custom_algorithm(ranks)
        assert (
            "number of supplied feature names does not match the number "
            "of ranks provided." in str(e.value)
        )


def test_1d_custom_ranks_incorrect_dimension():
    ranks = np.array([[1.0, -0.076], [-0.076, 1.0]])
    with pytest.raises(ValueError) as e:
        Rank1D(
            features=["Feature1", "Feature2"],
        ).feature_ranks_custom_algorithm(ranks)
        assert "Ranks must be 1-dimensional" in str(e.value)


@image_comparison(
    baseline_images=["shapiro_from_results_no_features"],
    extensions=["png"],
    remove_text=False,
)
def test_1d_from_results_no_features():
    ranks = np.array([0.96784323, 0.79394983, 0.47927427])
    Rank1D(orient="h", figsize=(13, 7)).feature_ranks_custom_algorithm(ranks)


def test_ax_1d():
    fig, ax = plt.subplots(1, 1)
    out_ax = Rank1D().feature_ranks(X)
    assert ax is not out_ax
    out_ax = Rank1D(ax=ax).feature_ranks(X)
    assert ax is out_ax


def test_unknown_algorithm_2d():
    with pytest.raises(ValueError) as e:
        Rank2D(algorithm="unknown").feature_ranks(X)
        assert "'unknown' is unrecognized ranking method" in str(e.value)


@image_comparison(
    baseline_images=["pearson_2d_iris"], extensions=["png"], remove_text=False
)
def test_2d_pearson():
    Rank2D(algorithm="pearson", features=features, figsize=(12, 12)).feature_ranks(X)


@image_comparison(
    baseline_images=["pearson_2d_iris_colormap"], extensions=["png"], remove_text=False
)
def test_2d_pearson_colormap():
    Rank2D(
        algorithm="pearson", features=features, figsize=(12, 12), colormap="PuBuGn"
    ).feature_ranks(X)


@image_comparison(
    baseline_images=["covariance_2d_iris"], extensions=["png"], remove_text=False
)
def test_2d_covariance():
    Rank2D(algorithm="covariance", features=features, figsize=(12, 12)).feature_ranks(X)


@image_comparison(
    baseline_images=["spearman_2d_iris"], extensions=["png"], remove_text=False
)
def test_2d_spearman():
    Rank2D(algorithm="spearman", features=features, figsize=(12, 12)).feature_ranks(X)


@image_comparison(
    baseline_images=["kendalltau_2d_iris"], extensions=["png"], remove_text=False
)
def test_2d_kendalltau():
    Rank2D(algorithm="KendallTau", features=features, figsize=(12, 12)).feature_ranks(X)


def test_2d_incorrect_features():
    with pytest.raises(ValueError) as e:

        Rank2D(
            features=["Sepal length", "Sepal width"], algorithm="pearson"
        ).feature_ranks(X)
        assert (
            "number of supplied feature names does not match the number "
            "of columns in the training data." in str(e.value)
        )


@image_comparison(
    baseline_images=["pearson_2d_iris_numpy"], extensions=["png"], remove_text=False
)
def test_2d_numpy():
    X_numpy = X.to_numpy()
    assert isinstance(X_numpy, np.ndarray)
    Rank2D(figsize=(7, 7)).feature_ranks(X_numpy)


@image_comparison(
    baseline_images=["pearson_2d_iris_numpy_features"],
    extensions=["png"],
    remove_text=False,
)
def test_2d_numpy_features():
    X_numpy = X.to_numpy()
    assert isinstance(X_numpy, np.ndarray)
    Rank2D(features=features, figsize=(12, 12)).feature_ranks(X_numpy)


@image_comparison(
    baseline_images=["custom_ranks_2d"], extensions=["png"], remove_text=False
)
def test_2d_custom_ranks():
    ranks = np.array(
        [
            [1.0, -0.076, 0.71, 0.65],
            [-0.076, 1.0, -0.18, -0.15],
            [0.71, -0.18, 1.0, 0.80],
            [0.65, -0.15, 0.80, 1.0],
        ]
    )
    Rank2D(
        features=["Feature1", "Feature2", "Feature3", "Feature4"], figsize=(9, 12)
    ).feature_ranks_custom_algorithm(ranks)


def test_2d_custom_ranks_incorrect_dimension():
    ranks = np.array([1.0, -0.076, 0.71, 0.65])
    with pytest.raises(ValueError) as e:
        Rank2D(
            features=["Feature1", "Feature2", "Feature3", "Feature4"], figsize=(9, 12)
        ).feature_ranks_custom_algorithm(ranks)
        assert "Ranks must be 2-dimensional" in str(e.value)


@image_comparison(
    baseline_images=["custom_ranks_2d_no_features"],
    extensions=["png"],
    remove_text=False,
)
def test_2d_custom_ranks_no_features():
    ranks = np.array(
        [
            [1.0, -0.076, 0.71, 0.65],
            [-0.076, 1.0, -0.18, -0.15],
            [0.71, -0.18, 1.0, 0.80],
            [0.65, -0.15, 0.80, 1.0],
        ]
    )
    Rank2D(figsize=(9, 12)).feature_ranks_custom_algorithm(ranks)


def test_2d_custom_ranks_incorrect_features():
    ranks = np.array(
        [
            [1.0, -0.076, 0.71, 0.65],
            [-0.076, 1.0, -0.18, -0.15],
            [0.71, -0.18, 1.0, 0.80],
            [0.65, -0.15, 0.80, 1.0],
        ]
    )
    with pytest.raises(ValueError) as e:
        Rank2D(
            features=["Feature1", "Feature2", "Feature3"]
        ).feature_ranks_custom_algorithm(ranks)
        assert (
            "number of supplied feature names does not match the number "
            "of ranks provided." in str(e.value)
        )


def test_ax_2d():
    fig, ax = plt.subplots(1, 1)
    out_ax = Rank2D().feature_ranks(X)
    assert ax is not out_ax
    out_ax = Rank2D(ax=ax).feature_ranks(X)
    assert ax is out_ax
