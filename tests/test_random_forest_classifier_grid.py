import pytest
from sklearn_evaluation.grid.classifier_grid import GridTypes
from sklearn_evaluation.grid import RandomForestClassifierGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

grid_types_str = ["tiny", "small", "medium", "large", "xlarge"]


@pytest.mark.parametrize(
    "grid, expected",
    [
        (grid_types_str[0], GridTypes.TINY),
        (grid_types_str[1], GridTypes.SMALL),
        (grid_types_str[2], GridTypes.MEDIUM),
        (grid_types_str[3], GridTypes.LARGE),
        (grid_types_str[4], GridTypes.X_LARGE),
    ],
)
def test_initialize_grid_model(grid, expected):
    model = RandomForestClassifierGrid(grid=grid)
    assert model.grid.value == expected.value
    assert isinstance(model.estimator_, RandomForestClassifier)
    assert isinstance(model.grid_search_cv_, GridSearchCV)
    assert isinstance(model.estimator_, RandomForestClassifier)


@pytest.mark.parametrize("grid", grid_types_str)
def test_fit(grid):
    model = RandomForestClassifierGrid(grid=grid)
    X, y = datasets.make_classification(
        200, 10, n_informative=5, class_sep=0.7, random_state=42
    )
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    model = RandomForestClassifierGrid(grid=grid)
    model.set_test_data(X_test, y_test)

    model.fit(X, y)

    assert isinstance(model.grid_search_cv_, GridSearchCV)


@pytest.mark.parametrize("grid", grid_types_str)
def test_set_test_data(grid):

    X, y = datasets.make_classification(
        200, 10, n_informative=5, class_sep=0.7, random_state=42
    )

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    model = RandomForestClassifierGrid(grid=grid)
    model.set_test_data(X_test, y_test)

    assert np.array_equal(model.X_test, X_test)
    assert np.array_equal(model.y_test, y_test)


@pytest.mark.parametrize("grid", grid_types_str)
@pytest.mark.parametrize(
    "X_test, y_test, expected_error_type, expected_error_message",
    [
        ([], [], ValueError, "X_test and y_test should not be empty"),
        ([], None, TypeError, "X_test and y_test should be array-like"),
        (None, [], TypeError, "X_test and y_test should be array-like"),
        (None, None, TypeError, "X_test and y_test should be array-like"),
        ([0, 1, 2], [0, 1], ValueError, "Found input variables with inconsistent"),
        ([0, 1], [0, 1, 2], ValueError, "Found input variables with inconsistent"),
    ],
)
def test_set_test_data_error(
    grid, X_test, y_test, expected_error_type, expected_error_message
):
    model = RandomForestClassifierGrid(grid=grid)

    with pytest.raises(expected_error_type) as excinfo:
        model.set_test_data(X_test, y_test)

    assert expected_error_message in str(excinfo.value)


@pytest.mark.parametrize("grid", grid_types_str)
def test_warning_when_no_test_data_was_given(grid):
    model = RandomForestClassifierGrid(grid=grid)
    X, y = datasets.make_classification(
        200, 10, n_informative=5, class_sep=0.7, random_state=42
    )

    model = RandomForestClassifierGrid(grid=grid)

    model.fit(X, y)

    with pytest.warns(UserWarning, match="no test data was provided"):
        model.confusion_matrix()

    with pytest.warns(UserWarning, match="no test data was provided"):
        model.roc()

    assert isinstance(model.grid_search_cv_, GridSearchCV)


# TODO: Add test for param_grids values
# (check if grid_search_cv_ initialized with relevant param_grid)
# TODO: Based on the param_grids test the plots


def test_confusion_matrix():
    pass


def test_roc():
    pass


def test_feature_importances():
    pass


def test_grid_search():
    pass
