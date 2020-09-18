from pathlib import Path

import pytest
from matplotlib.testing.conftest import (mpl_test_settings,
                                         mpl_image_comparison_parameters,
                                         pytest_configure, pytest_unconfigure,
                                         pd)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import datasets


def _path_to_tests():
    return Path(__file__).absolute().parent


@pytest.fixture(scope='session')
def path_to_tests():
    return _path_to_tests()


@pytest.fixture(scope='session')
def grid_search_3_params():
    parameters = {
        'n_estimators': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
    }

    est = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(est, parameters, cv=5)

    X, y = datasets.make_classification(200, 10, n_informative=5, class_sep=0.7,
                                        random_state=42)
    clf.fit(X, y)

    return clf


@pytest.fixture(scope='session')
def grid_search_4_params():
    parameters = {
        'n_estimators': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2],
    }

    est = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(est, parameters, cv=5)

    X, y = datasets.make_classification(200, 10, n_informative=5, class_sep=0.7,
                                        random_state=42)
    clf.fit(X, y)

    return clf


@pytest.fixture(scope='session')
def grid_search_2_params():
    parameters = {
        'n_estimators': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy'],
    }

    est = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(est, parameters, cv=5)

    X, y = datasets.make_classification(200, 10, n_informative=5, class_sep=0.7,
                                        random_state=42)
    clf.fit(X, y)

    return clf
