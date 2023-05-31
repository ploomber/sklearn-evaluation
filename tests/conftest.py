from copy import copy
import sys
import os
from pathlib import Path

import pytest
import numpy as np

# These are fixtures to get the same configuration that matplotlib uses
# to run tests with pytest. Note that importing other fixtures from that
# module leads to weird error messages (e.g. "pd", which patches pandas)
# https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/testing/conftest.py
from matplotlib.testing.conftest import mpl_test_settings  # noqa
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
def ploomber_value_error_message():
    COMMUNITY = (
        "\nIf you need help solving this "
        "issue, send us a message: https://ploomber.io/community"
    )

    return COMMUNITY


@pytest.fixture(scope="session")
def precision_recall_binary_classification():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0])
    y_score = np.array(
        [
            [0.58, 0.42],
            [0.69, 0.38],
            [0.62, 0.38],
            [0.32, 0.68],
            [0.17, 0.83],
            [0.19, 0.81],
            [0.85, 0.15],
        ]
    )
    return y_true, y_score


@pytest.fixture(scope="session")
def precision_recall_multiclass_classification_set_one():
    y_true = np.array([1, 1, 2, 0, 0])
    y_score = np.array(
        [
            [0.18, 0.43, 0.39],
            [0.37, 0.48, 0.15],
            [0.19, 0.17, 0.64],
            [0.29, 0.11, 0.6],
            [0.19, 0.12, 0.69],
        ]
    )
    return y_true, y_score


@pytest.fixture(scope="session")
def precision_recall_multiclass_classification_set_two():
    y_true = np.array([1, 1, 2, 0, 0])
    y_score = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return y_true, y_score


@pytest.fixture(scope="session")
def calibration_curve_raw_data():
    X, y = datasets.make_classification(
        n_samples=7000, n_features=2, n_informative=2, n_redundant=0, random_state=0
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0
    )

    rf = RandomForestClassifier(random_state=0)
    lr = LogisticRegression(random_state=0)

    rf_probas = rf.fit(X_train, y_train).predict_proba(X_test)
    lr_probas = lr.fit(X_train, y_train).predict_proba(X_test)

    probabilities = [rf_probas, lr_probas]

    clf_names = ["Random Forest", "Logistic Regression"]

    return y_test, probabilities, clf_names


@pytest.fixture(scope="session")
def calibration_curve_plot_metrics_data():
    mean_predicted_value_one = [
        [0.02024768, 0.29093366, 0.46554422, 0.68282258, 0.91699482]
    ]

    fraction_of_positives_one = [
        [0.06147722, 0.33673219, 0.57870748, 0.79467742, 0.94041451]
    ]

    clf_names_one = ["Random Forest"]

    mean_predicted_value_two = [
        [0.03392278, 0.3362952, 0.50780503, 0.81189161, 0.89122685]
    ]

    fraction_of_positives_two = [
        [0.04212573, 0.39193548, 0.51515152, 0.805575345, 0.99522876]
    ]

    clf_names_two = ["Logistic Regression"]

    return (
        mean_predicted_value_one,
        fraction_of_positives_one,
        clf_names_one,
        mean_predicted_value_two,
        fraction_of_positives_two,
        clf_names_two,
    )
