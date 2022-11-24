import pytest
import numpy as np
from sklearn_evaluation import plot, __version__

import warnings


@pytest.fixture
def y():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    return y_true, y_pred


@pytest.mark.parametrize(
    "target_names, target_names_expected",
    [
        [None, ["0", "1"]],
        [["A", "B"], ["A", "B"]],
    ],
)
def test_from_raw_data(y, target_names, target_names_expected):
    y_true, y_pred = y
    cm = plot.ClassificationReport.from_raw_data(
        y_true, y_pred, target_names=target_names
    )

    assert cm._get_data() == {
        "class": "sklearn_evaluation.plot.ClassificationReport",
        "matrix": [[0.5, 0.5, 0.5, 4.0], [0.5, 0.5, 0.5, 4.0]],
        "keys": ["precision", "recall", "f1-score", "support"],
        "target_names": target_names_expected,
        "version": __version__,
    }


def test_dump(tmp_directory, y):
    y_true, y_pred = y
    cm = plot.ClassificationReport.from_raw_data(y_true, y_pred)

    cm.dump("cr.json")
    cm2 = plot.ClassificationReport.from_dump("cr.json")
    assert cm._get_data() == cm2._get_data()


def test_warning(y):
    y_true, y_pred = y

    with pytest.warns(
        FutureWarning, match="ClassificationReport will change its signature"
    ):
        plot.ClassificationReport(y_true, y_pred)


def test_raw_data_doesnt_warn(y):
    y_true, y_pred = y

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot.ClassificationReport.from_raw_data(y_true, y_pred)
