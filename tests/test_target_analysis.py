"""
Some of the test cases are based on the yellowbricks test module.

License below

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

import pytest

import numpy as np
import pandas as pd
from sklearn import datasets

from matplotlib.testing.decorators import image_comparison

from sklearn.model_selection import train_test_split

from sklearn_evaluation import plot


iris_data = datasets.load_iris()


def test_invalid_target():
    """
    A value error should be raised on invalid train or test target
    """
    y_valid = np.random.randint(2, size=100)
    y_invalid = np.random.uniform(size=100)
    with pytest.raises(TypeError):
        plot.target_analysis(y_invalid)
    with pytest.raises(TypeError):
        plot.target_analysis(y_valid, y_invalid)


def test_class_names_must_match(target_analysis_binary, ploomber_value_error_message):
    """
    Assert error raised when more classes are in data than specified
    """

    _, _, y_train, y_test = target_analysis_binary

    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.target_analysis(y_train, y_test=y_test, labels=["a", "b", "c"])


@image_comparison(
    baseline_images=["binary_balance"], extensions=["png"], remove_text=False
)
def test_binary_balance(target_analysis_binary):
    """
    Test binary classification in balance mode
    """

    _, _, y_train, y_test = target_analysis_binary
    plot.target_analysis(y_train)


@image_comparison(
    baseline_images=["binary_compare"], extensions=["png"], remove_text=False
)
def test_binary_compare(target_analysis_binary):
    """
    Test binary classification in compare mode
    """

    _, _, y_train, y_test = target_analysis_binary
    plot.target_analysis(y_train, y_test)


@image_comparison(
    baseline_images=["multiclass_balance"], extensions=["png"], remove_text=False
)
def test_multiclass_balance(target_analysis_multiclass):
    """
    Test multiclass classification in balance mode
    """
    _, _, y_train, y_test = target_analysis_multiclass
    plot.target_analysis(y_train)


@image_comparison(
    baseline_images=["multiclass_compare"], extensions=["png"], remove_text=False
)
def test_multiclass_compare(target_analysis_multiclass):
    """
    Test multiclass classification in compare mode
    """
    _, _, y_train, y_test = target_analysis_multiclass
    plot.target_analysis(y_train, y_test)


@image_comparison(
    baseline_images=["multiclass_compare_iris_pandas"],
    extensions=["png"],
    remove_text=False,
)
def test_multiclass_compare_iris_pandas():
    """
    Test multiclass classification in compare mode for pandas input type
    """
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df["target"] = pd.Series(iris_data.target)
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    plot.target_analysis(train.target, test.target)


@image_comparison(
    baseline_images=["iris_labels_compare"], extensions=["png"], remove_text=False
)
def test_labels_iris():
    X = iris_data.data
    y = iris_data.target
    _, _, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    plot.target_analysis(y_train, y_test, labels=["Setosa", "Versicolour", "Virginica"])


@image_comparison(baseline_images=["bar_color"], extensions=["png"], remove_text=False)
def test_bar_color(target_analysis_binary):
    """
    Test multiclass classification in compare mode
    """
    _, _, y_train, y_test = target_analysis_binary
    plot.target_analysis(y_train, y_test, colors=["y", "g"])
