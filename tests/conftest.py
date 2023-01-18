from copy import copy
import sys
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
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


@pytest.fixture(scope="session")
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=101
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=101
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def roc_multi_classification_raw_data(target_analysis_multiclass):
    X_train, X_test, y_train, y_test = target_analysis_multiclass
    classifier = LogisticRegression()
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    return y_test, y_score


@pytest.fixture
def roc_multi_classification_raw_data_set2():
    from sklearn.datasets import load_iris

    iris = load_iris()
    X, y = iris.data, iris.target
    y = iris.target_names[y]

    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape

    X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

    classifier = LogisticRegression()
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    return y_test, y_score


@pytest.fixture
def regression_data():
    from sklearn.linear_model import LinearRegression

    X, y = datasets.make_regression(
        n_samples=2000, n_features=4, n_informative=3, noise=30.0, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred


@pytest.fixture
def roc_multi_classification_values():
    roc_rates_n_classes = [
        {
            "fpr": [
                0.0,
                0.0,
                0.11764705882352941,
                0.11764705882352941,
                0.23529411764705882,
                0.23529411764705882,
                1.0,
            ],
            "tpr": [
                0.0,
                0.3333333333333333,
                0.3333333333333333,
                0.6666666666666666,
                0.6666666666666666,
                1.0,
                1.0,
            ],
        },
        {
            "fpr": [
                0.0,
                0.0,
                0.058823529411764705,
                0.058823529411764705,
                0.17647058823529413,
                0.17647058823529413,
                1.0,
            ],
            "tpr": [
                0.0,
                0.3333333333333333,
                0.3333333333333333,
                0.6666666666666666,
                0.6666666666666666,
                1.0,
                1.0,
            ],
        },
        {
            "fpr": [
                0.0,
                0.0,
                0.0,
                0.26666666666666666,
                0.26666666666666666,
                0.7333333333333333,
                0.7333333333333333,
                1.0,
            ],
            "tpr": [0.0, 0.2, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0],
        },
        {
            "fpr": [0.0, 0.0, 0.0, 0.13333333333333333, 0.13333333333333333, 1.0],
            "tpr": [0.0, 0.2, 0.4, 0.4, 1.0, 1.0],
        },
        {"fpr": [0.0, 0.0, 0.0, 1.0], "tpr": [0.0, 0.25, 1.0, 1.0]},
    ]

    avg_fpr = [
        0.0,
        0.0,
        0.0,
        0.0125,
        0.0125,
        0.0375,
        0.0375,
        0.0625,
        0.0625,
        0.075,
        0.075,
        0.0875,
        0.0875,
        0.125,
        0.125,
        0.15,
        0.15,
        0.2125,
        0.2125,
        0.35,
        0.35,
        0.6,
        0.6,
        1.0,
    ]
    avg_tpr = [
        0.0,
        0.05,
        0.35,
        0.35,
        0.4,
        0.4,
        0.5,
        0.5,
        0.55,
        0.55,
        0.6,
        0.6,
        0.65,
        0.65,
        0.7,
        0.7,
        0.75,
        0.75,
        0.8,
        0.8,
        0.95,
        0.95,
        1.0,
        1.0,
    ]

    fpr = [avg_fpr]
    tpr = [avg_tpr]
    labels = ["micro-average ROC curve"]

    for i in range(len(roc_rates_n_classes)):
        fpr_tpr = roc_rates_n_classes[i]
        fpr.append(fpr_tpr["fpr"])
        tpr.append(fpr_tpr["tpr"])
        labels.append(f"(class {i}) ROC curve")

    return fpr, tpr, labels


@pytest.fixture
def roc_values():
    fpr = [0.0, 0.2, 0.4, 0.4, 0.6, 1.0]
    tpr = [0.0, 0.2, 0.4, 1.0, 1.0, 1.0]

    return fpr, tpr


@pytest.fixture
def ploomber_value_error_message():
    COMMUNITY = (
        "\nIf you need help solving this "
        "issue, send us a message: https://ploomber.io/community"
    )

    return COMMUNITY
