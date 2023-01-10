import pytest
import numpy as np
from sklearn_evaluation import plot, __version__


@pytest.fixture
def y():
    y_test = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_score = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.1, 0.1, 0.3],
        ]
    ).T
    return y_test, y_score


@pytest.mark.parametrize(
    "fpr, tpr, expected_fpr, expected_tpr",
    [
        ([0.0, 0.2, 0.4], [0.0, 0.2, 0.4], [[0.0, 0.2, 0.4]], [[0.0, 0.2, 0.4]]),
        (
            [[0.0, 0.2], [0.0, 0.4]],
            [[0.0, 0.2], [0.0, 0.5]],
            [[0.0, 0.2], [0.0, 0.4]],
            [[0.0, 0.2], [0.0, 0.5]],
        ),
        (np.array([0.0, 0.2]), np.array([0.0, 0.2]), [[0.0, 0.2]], [[0.0, 0.2]]),
    ],
)
def test_get_data(fpr, tpr, expected_fpr, expected_tpr):
    roc = plot.ROC(fpr, tpr)

    assert roc._get_data() == {
        "class": "sklearn_evaluation.plot.ROC",
        "version": __version__,
        "fpr": expected_fpr,
        "tpr": expected_tpr,
    }

    assert roc._get_data()["fpr"] == roc.fpr
    assert roc._get_data()["tpr"] == roc.tpr


def test_from_raw_data(y):
    y_test, y_score = y
    roc = plot.ROC.from_raw_data(y_test, y_score)

    assert roc._get_data() == {
        "class": "sklearn_evaluation.plot.ROC",
        "version": __version__,
        "fpr": roc.fpr,
        "tpr": roc.tpr,
    }


def test_from_raw_data_multi(tmp_directory, y, roc_multi_classification_raw_data):
    y_test, y_score = roc_multi_classification_raw_data

    roc = plot.ROC.from_raw_data(y_test, y_score)

    assert roc._get_data() == {
        "class": "sklearn_evaluation.plot.ROC",
        "version": __version__,
        "fpr": roc.fpr,
        "tpr": roc.tpr,
    }


def test_roc_dump(tmp_directory, y):
    y_test, y_score = y
    roc = plot.ROC.from_raw_data(y_test, y_score)

    roc.dump("roc.json")
    roc2 = plot.ROC.from_dump("roc.json")
    assert roc._get_data() == roc2._get_data()


def test_roc_dump_multi(tmp_directory, roc_multi_classification_raw_data):
    y_test, y_score = roc_multi_classification_raw_data

    roc = plot.ROC.from_raw_data(y_test, y_score)

    roc.dump("roc.json")

    roc2 = plot.ROC.from_dump("roc.json")

    assert roc._get_data() == roc2._get_data()


def test_roc_sub_not_implemented_error(roc_values):
    fpr, tpr = roc_values

    roc = plot.ROC(fpr, tpr)

    with pytest.raises(NotImplementedError) as excinfo:
        roc.__sub__()
        assert "Not applicable for ROC" in str(excinfo.value)


@pytest.mark.parametrize(
    "fpr, tpr",
    [
        ([0.0, 0.2, 0.4, 0.4, 0.6, 1.0], []),
        ([], [0.0, 0.2, 0.4, 1.0, 1.0, 1.0]),
        ([], []),
    ],
)
def test_none_fpr_or_tpr_error(fpr, tpr, ploomber_value_error_message):

    with pytest.raises(ValueError, match=ploomber_value_error_message) as excinfo:
        plot.ROC(fpr, tpr)
    assert "fpr and tpr must not be empty" in str(excinfo.value)


@pytest.mark.parametrize(
    "fpr, tpr",
    [
        ([0.0, 0.2, 0.4, 0.4, 0.6, 1.0], None),
        (None, [0.0, 0.2, 0.4, 1.0, 1.0, 1.0]),
        (None, None),
    ],
)
def test_empty_fpr_or_tpr_error(fpr, tpr):

    with pytest.raises(TypeError) as excinfo:
        plot.ROC(fpr, tpr)
    assert "fpr and tpr must be defined" in str(excinfo.value)


@pytest.mark.parametrize(
    "fpr, tpr",
    [
        ([0.0, 0.2, 0.4, 0.4, 0.6, 1.0], [0.0, 0.2, 0.4, 0.4, 0.6]),
        ([0.0, 0.2, 0.4, 0.4, 0.6], [0.0, 0.2, 0.4, 1.0, 1.0, 1.0]),
    ],
)
def test_fpr_and_tpr_different_lengths(fpr, tpr, ploomber_value_error_message):
    with pytest.raises(ValueError, match=ploomber_value_error_message) as excinfo:
        plot.ROC(fpr, tpr)

    assert (
        "fpr and tpr lengths should correspond. "
        f"Recevied: fpr {len(fpr)} != tpr {len(tpr)}" in str(excinfo.value)
    )


@pytest.mark.parametrize(
    "fpr, tpr",
    [
        ([0.0, 0.2, 0.4, 0.4, 0.6, 1.0], np.array([0.0, 0.2, 0.4, 0.4, 0.6])),
        (np.array([0.0, 0.2, 0.4, 0.4, 0.6]), [0.0, 0.2, 0.4, 1.0, 1.0, 1.0]),
        ([[0.0, 0.2], [0.0, 0.4]], np.array([[0.0, 0.2], [0.0, 0.5]])),
    ],
)
def test_fpr_and_tpr_different_types(fpr, tpr):
    with pytest.raises(TypeError) as excinfo:
        plot.ROC(fpr, tpr)

    assert (
        "fpr and tpr must be the same type. "
        f"Recevied: fpr {type(fpr)} != tpr {type(tpr)}" in str(excinfo.value)
    )
