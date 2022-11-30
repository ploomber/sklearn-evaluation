"""
Test cases for ks statistics

NOTE: this is largely based in the scikit-plot metrics module. License below.

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

import sys
import numpy as np
import pytest
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison as _image_comparison
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn_evaluation.plot import ks_statistic

# older versions of Python are not compatible with the latest version of
# matplotlib, which leads to image differences. We increase the tolerance in
# such cases
image_comparison = partial(_image_comparison,
                           tol=21 if sys.version_info.minor==8 else 0)

X, y = load_breast_cancer(return_X_y=True)


def _convert_labels_into_string(y_true):
    return ["A" if x == 0 else x for x in y_true]


@image_comparison(baseline_images=['string_classes_ks_statistics'],
                  extensions=['png'],
                  remove_text=False)
def test_string_classes():
    clf = LogisticRegression(random_state=101)
    clf.fit(X, _convert_labels_into_string(y))
    probas = clf.predict_proba(X)
    ks_statistic(_convert_labels_into_string(y), probas)


def test_two_classes():
    np.random.seed(0)
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression()
    clf.fit(X, y)
    probas = clf.predict_proba(X)
    with pytest.raises(ValueError) as e:
        ks_statistic(y, probas)
    assert "Cannot calculate KS statistic for data with 3 category/ies" in str(
        e.value)


def test_ax():
    np.random.seed(0)
    clf = LogisticRegression()
    clf.fit(X, y)
    probas = clf.predict_proba(X)
    fig, ax = plt.subplots(1, 1)
    out_ax = ks_statistic(y, probas)
    assert ax is not out_ax
    out_ax = ks_statistic(y, probas, ax=ax)
    assert ax is out_ax


@image_comparison(baseline_images=[
    'array_like_ks_statistics_1', 'array_like_ks_statistics_2',
    'array_like_ks_statistics_3'
],
                  extensions=['png'],
                  remove_text=False)
def test_array_like():
    ks_statistic([0, 1], [[0.8, 0.2], [0.2, 0.8]])
    ks_statistic([0, 'a'], [[0.8, 0.2], [0.2, 0.8]])
    ks_statistic(['b', 'a'], [[0.8, 0.2], [0.2, 0.8]])
