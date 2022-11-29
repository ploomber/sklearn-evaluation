"""
Test cases for cumulative gain and lift curve

NOTE: this is largely based in the scikit-plot test module. License below.

MIT License

Copyright (c) [2018] [Reiichiro Nakano]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn_evaluation.plot import plot_cumulative_gain, plot_lift_curve

X, y = load_breast_cancer(return_X_y=True)


def _convert_labels_into_string(y_true):
    return ["A" if x == 0 else x for x in y_true]


@image_comparison(baseline_images=['string_classes_cumulative_gain'],
                  extensions=['png'],
                  remove_text=False)
def test_string_classes_cumulative_gain():
    clf = LogisticRegression()
    clf.fit(X, _convert_labels_into_string(y))
    probas = clf.predict_proba(X)
    plot_cumulative_gain(_convert_labels_into_string(y), probas)


def test_two_classes_cumulative_gain():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression()
    clf.fit(X, y)
    probas = clf.predict_proba(X)
    with pytest.raises(ValueError) as e:
        plot_cumulative_gain(y, probas)
    assert "Cannot calculate Cumulative Gains for data with 3 category/ies" in str(
        e.value)


def test_ax_cumulative_gain():
    clf = LogisticRegression()
    clf.fit(X, y)
    probas = clf.predict_proba(X)
    fig, ax = plt.subplots(1, 1)
    out_ax = plot_cumulative_gain(y, probas)
    assert ax is not out_ax
    out_ax = plot_cumulative_gain(y, probas, ax=ax)
    assert ax is out_ax


@image_comparison(baseline_images=['array_like_cumulative_gain_1',
                                   'array_like_cumulative_gain_2',
                                   'array_like_cumulative_gain_3'],
                  extensions=['png'],
                  remove_text=False)
def test_array_like_cumulative_gain():
    plot_cumulative_gain([0, 1], [[0.8, 0.2], [0.2, 0.8]])
    plot_cumulative_gain([0, 'a'], [[0.8, 0.2], [0.2, 0.8]])
    plot_cumulative_gain(['b', 'a'], [[0.8, 0.2], [0.2, 0.8]])


@image_comparison(baseline_images=['string_classes_lift_curve'],
                  extensions=['png'],
                  remove_text=False)
def test_string_classes_lift_curve():
    clf = LogisticRegression()
    clf.fit(X, _convert_labels_into_string(y))
    probas = clf.predict_proba(X)
    plot_lift_curve(_convert_labels_into_string(y), probas)


def test_two_classes_lift_curve():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression()
    clf.fit(X, y)
    probas = clf.predict_proba(X)
    with pytest.raises(ValueError) as e:
        plot_lift_curve(y, probas)
    assert "Cannot calculate Lift Curve for data with 3 category/ies" in str(
        e.value)


def test_ax_lift_curve():
    clf = LogisticRegression()
    clf.fit(X, y)
    probas = clf.predict_proba(X)
    fig, ax = plt.subplots(1, 1)
    out_ax = plot_lift_curve(y, probas)
    assert ax is not out_ax
    out_ax = plot_lift_curve(y, probas, ax=ax)
    assert ax is out_ax


def test_array_like_lift_curve():
    plot_lift_curve([0, 1], [[0.8, 0.2], [0.2, 0.8]])
    #plot_lift_curve([0, 'a'], [[0.8, 0.2], [0.2, 0.8]])
    #plot_lift_curve(['b', 'a'], [[0.8, 0.2], [0.2, 0.8]])
