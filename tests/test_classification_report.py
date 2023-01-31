import pytest
import numpy as np
from sklearn_evaluation import plot, __version__


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


def test_plot_classification_report_bad_input_value_error(
    y, ploomber_value_error_message
):
    _, y_pred = y

    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.ClassificationReport.from_raw_data([1, 2, 1], y_pred)

    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.classification_report([1, 2, 1], y_pred, target_names=["Not spam", "Spam"])
