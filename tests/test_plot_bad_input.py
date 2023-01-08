import pytest
import numpy as np

from sklearn_evaluation import plot

# todo: consider adding values from test_plot_input.py


@pytest.fixture
def y():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    return y_true, y_pred


def test_plot_confusion_matrixe_bad_input_value_error(ploomber_value_error_message):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.confusion_matrix(None, [1, 0])


def test_plot_roc_bad_input_value_error(ploomber_value_error_message):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.roc([1, 0], None)

    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.ROC([1, 0], None)

    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.ROC.from_raw_data([1, 0], None)


def test_plot_precision_recalle_bad_input_value_error(ploomber_value_error_message):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.precision_recall(None, None)


def test_plot_precision_at_proportionse_bad_input_value_error(
    ploomber_value_error_message,
):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.precision_at_proportions(None, [1, 0])


@pytest.mark.parametrize("bad_y_true", [np.array([0, 0]), [""]])
def test_plot_calibration_curve_bad_input_value_error(
    bad_y_true, ploomber_value_error_message
):
    from examples.calibration_curve import probabilities, clf_names

    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.calibration_curve(bad_y_true, probabilities, clf_names)


def test_plot_feature_importances_bad_input_value_error(ploomber_value_error_message):
    from examples.feature_importances import model

    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.feature_importances(model, top_n=-1)

    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.feature_importances(None)


@pytest.mark.parametrize(
    "bad_train_sizes", [[], [143, 467, 790, 1113, 1437, 1], [1, 0], ""]
)
def test_plot_learning_curve_bad_input_value_error(
    bad_train_sizes, ploomber_value_error_message
):
    from examples.learning_curve import train_scores, test_scores

    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.learning_curve(train_scores, test_scores, bad_train_sizes)


def test_plot_metrics_at_thresholds_bad_input_value_error(
    y, ploomber_value_error_message
):
    y_true, y_pred = y

    def fn(a, b):
        return [1, 2, 3]

    with pytest.raises(ValueError, match=ploomber_value_error_message) as e:
        plot.metrics_at_thresholds(fn, y_true, y_pred)

    assert "x and y must have same first dimension" in str(e.value)


@pytest.mark.parametrize("bad_y_true_value", [None, [], [1, 2, 3, 4, 5], [1, 0], ""])
def test_plot_prediction_error_bad_input_value_error(
    bad_y_true_value, y, ploomber_value_error_message
):
    _, y_pred = y

    with pytest.raises(ValueError, match=ploomber_value_error_message) as e:
        plot.prediction_error(np.array(bad_y_true_value), y_pred)

    assert "parameters should have same shape" in str(e.value)


@pytest.mark.parametrize(
    "bad_y_pairs", [([0, 1], [0]), ([0], [1, 1]), ([], [1]), ([1, 2, 3], [])]
)
def test_plot_residuals_bad_input_value_error(
    bad_y_pairs, ploomber_value_error_message
):
    y_true, y_pred = bad_y_pairs

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    with pytest.raises(ValueError, match=ploomber_value_error_message) as e:
        plot.residuals(y_true, y_pred)

    assert "parameters should have same shape" in str(e.value)


@pytest.mark.parametrize("bad_y_scores", [None, [(1, 2)]])
def test_plot_scores_distribution_bad_input_value_error(
    bad_y_scores, ploomber_value_error_message
):
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.scores_distribution(bad_y_scores, n_bins=10)
    assert "y should be a 1d array, got an array of shape"


def test_plot_validation_curve_bad_input_value_error(ploomber_value_error_message):
    from examples.validation_curve import (
        train_scores,
        test_scores,
        param_range,
        param_name,
    )

    bad_train_scores = train_scores[:2]
    with pytest.raises(ValueError, match=ploomber_value_error_message) as e:
        plot.validation_curve(
            bad_train_scores, test_scores, param_range, param_name, semilogx=True
        )

    assert "x and y must have same first dimension" in str(e.value)


@pytest.mark.parametrize("bad_y", [None, []])
def test_plot_cooks_distance_bad_input_value_error(bad_y, ploomber_value_error_message):
    from examples.cooks_distance import X

    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.cooks_distance(X, bad_y)
