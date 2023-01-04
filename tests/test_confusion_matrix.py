import pytest
import numpy as np
from sklearn_evaluation import plot, __version__

import warnings
from ploomber_core.exceptions import PloomberValueError


@pytest.fixture
def y():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    return y_true, y_pred


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "target_names, target_names_expected",
    [
        [None, ["Class 0", "Class 1"]],
        [["A", "B"], ["A", "B"]],
    ],
)
def test_from_raw_data(y, normalize, target_names, target_names_expected):
    y_true, y_pred = y
    cm = plot.ConfusionMatrix.from_raw_data(
        y_true, y_pred, normalize=normalize, target_names=target_names
    )

    denominator = 4 if normalize else 1
    value = 2 / denominator

    assert cm._get_data() == {
        "class": "sklearn_evaluation.plot.ConfusionMatrix",
        "cm": [[value, value], [value, value]],
        "normalize": normalize,
        "target_names": target_names_expected,
        "version": __version__,
    }


def test_dump(tmp_directory, y):
    y_true, y_pred = y
    cm = plot.ConfusionMatrix.from_raw_data(y_true, y_pred)

    cm.dump("cm.json")
    cm2 = plot.ConfusionMatrix.from_dump("cm.json")
    assert cm._get_data() == cm2._get_data()


def test_warning(y):
    y_true, y_pred = y

    with pytest.warns(FutureWarning, match="ConfusionMatrix will change its signature"):
        plot.ConfusionMatrix(y_true, y_pred)


def test_raw_data_doesnt_warn(y):
    y_true, y_pred = y

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot.ConfusionMatrix.from_raw_data(y_true, y_pred)


@pytest.mark.parametrize("bad_y_true_value", [None, [], [1, 2, 3, 4, 5], [1, 0], ""])
def test_confusion_matrix_bad_input_value_error(y, bad_y_true_value):
    _, y_pred = y
    with pytest.raises(PloomberValueError):
        plot.ConfusionMatrix(bad_y_true_value, y_pred)
