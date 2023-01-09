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


def test_from_raw_data(y):
    y_test, y_score = y
    roc = plot.ROC.from_raw_data(y_test, y_score)

    assert roc._get_data() == {
        'class': 'sklearn_evaluation.plot.ROC',
        'version': __version__,
        'fpr': roc.fpr.tolist(),
        'tpr': roc.tpr.tolist(),
        'roc_rates_n_classes': roc.roc_rates_n_classes
    }


def test_from_raw_data_multi(tmp_directory, y, roc_multi_classification_values):
    y_test, y_score, _ = roc_multi_classification_values

    roc = plot.ROC(y_test, y_score)

    assert roc._get_data() == {
        'class': 'sklearn_evaluation.plot.ROC',
        'version': __version__,
        'fpr': roc.fpr.tolist(),
        'tpr': roc.tpr.tolist(),
        'roc_rates_n_classes': roc.roc_rates_n_classes
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


def test_roc_warning(y):
    y_test, y_score = y

    with pytest.warns(
        FutureWarning, match="ROC will change its signature"
    ):
        plot.roc(y_test, y_score)


def test_roc_sub_not_implemented_error(roc_values):
    fpr, tpr, _ = roc_values

    roc = plot.ROC(fpr, tpr)

    with pytest.raises(NotImplementedError) as excinfo:
        roc.__sub__()
        assert "Not applicable for ROC" in str(excinfo.value)
