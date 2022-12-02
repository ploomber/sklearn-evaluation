from copy import copy
import sys
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# These are fixtures to get the same configuration that matplotlib uses
# to run tests with pytest. Note that importing other fixtures from that
# module leads to weird error messages (e.g. "pd", which patches pandas)
# https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/testing/conftest.py
from matplotlib.testing.conftest import (  # noqa
    mpl_test_settings,
    mpl_image_comparison_parameters,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split


def _path_to_tests():
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def path_to_tests():
    return _path_to_tests()


@pytest.fixture(scope="session")
def path_to_tests_static():
    return _path_to_tests() / "static"


@pytest.fixture()
def tmp_directory(tmp_path):
    old = os.getcwd()
    os.chdir(str(tmp_path))
    yield str(Path(tmp_path).resolve())
    os.chdir(old)


@pytest.fixture
def add_current_to_sys_path():
    old = copy(sys.path)
    sys.path.insert(0, os.path.abspath("."))
    yield sys.path
    sys.path = old


@pytest.fixture(scope="session")
def grid_search_3_params():
    parameters = {
        "n_estimators": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2"],
    }

    est = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(est, parameters, cv=5)

    X, y = datasets.make_classification(
        200, 10, n_informative=5, class_sep=0.7, random_state=42
    )
    clf.fit(X, y)

    return clf


@pytest.fixture(scope="session")
def grid_search_4_params():
    parameters = {
        "n_estimators": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2],
    }

    est = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(est, parameters, cv=5)

    X, y = datasets.make_classification(
        200, 10, n_informative=5, class_sep=0.7, random_state=42
    )
    clf.fit(X, y)

    return clf


@pytest.fixture(scope="session")
def grid_search_2_params():
    parameters = {
        "n_estimators": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],
    }

    est = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(est, parameters, cv=5)

    X, y = datasets.make_classification(
        200, 10, n_informative=5, class_sep=0.7, random_state=42
    )
    clf.fit(X, y)

    return clf


@pytest.fixture(scope="session")
def grid_search_param_with_none():
    parameters = {
        "max_depth": [2, None],
        "criterion": ["gini", "entropy"],
    }

    est = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(est, parameters, cv=5)

    X, y = datasets.make_classification(
        200, 10, n_informative=5, class_sep=0.7, random_state=42
    )
    clf.fit(X, y)

    return clf


@pytest.fixture(scope='session')
def target_analysis_binary():
    kwargs = {
        "n_samples": 100,
        "n_classes": 2,
        "n_features": 20,
        "n_informative": 8,
        "n_redundant": 2,
        "n_clusters_per_class": 1,
        "random_state": 89092,
    }

    X, y = datasets.make_classification(**kwargs)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope='session')
def target_analysis_multiclass():
    kwargs = {
        "n_samples": 100,
        "n_classes": 5,
        "n_features": 20,
        "n_informative": 8,
        "n_redundant": 2,
        "n_clusters_per_class": 1,
        "random_state": 89092,
    }

    X, y = datasets.make_classification(**kwargs)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    return X_train, X_test, y_train, y_test
