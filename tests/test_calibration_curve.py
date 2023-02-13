"""
Tests for calibration curve
"""
import pytest
import numpy as np
from matplotlib.testing.decorators import image_comparison

from sklearn_evaluation import plot


mean_predicted_value = [
    [0.02024768, 0.29093366, 0.46554422, 0.68282258, 0.91699482],
    [0.03392278, 0.3362952, 0.50780503, 0.81189161, 0.89122685],
    [0.05405794, 0.35022914, 0.45182226, 0.85404048, 0.95437306],
]

fraction_of_positives = [
    [0.06147722, 0.33673219, 0.57870748, 0.79467742, 0.94041451],
    [0.04212573, 0.39193548, 0.51515152, 0.805575345, 0.99522876],
    [0.03518519, 0.31604356, 0.52941176, 0.78419753, 0.91847826],
]

clf_names = ["Random Forest", "Gaussian Naive Bayes", "Logistic Regression"]


@pytest.mark.parametrize(
    "mean_predicted_value, fraction_of_positives",
    [
        (mean_predicted_value, fraction_of_positives),
        (np.array(mean_predicted_value), np.array(fraction_of_positives)),
    ],
)
@image_comparison(
    baseline_images=["calibration_curve"], extensions=["png"], remove_text=True
)
def test_calibration_curve(mean_predicted_value, fraction_of_positives):
    plot.CalibrationCurve(
        mean_predicted_value, fraction_of_positives, label=clf_names
    ).plot()


@pytest.mark.parametrize(
    "mean_predicted_value, fraction_of_positives",
    [
        (mean_predicted_value, fraction_of_positives),
        (np.array(mean_predicted_value), np.array(fraction_of_positives)),
    ],
)
@image_comparison(
    baseline_images=["calibration_curve_default_label"],
    extensions=["png"],
    remove_text=True,
)
def test_calibration_curve_default_label(mean_predicted_value, fraction_of_positives):
    plot.CalibrationCurve(mean_predicted_value, fraction_of_positives).plot()


def test_metrics_length_mismatch(ploomber_value_error_message):
    with pytest.raises(ValueError) as e:
        plot.CalibrationCurve([0.5, 0.9], [0.6]).plot()
    assert (
        "mean_predicted_value and fraction_of_positives "
        "lengths should correspond" in str(e.value)
    )


@image_comparison(
    baseline_images=["from_raw_data"],
    extensions=["png"],
    remove_text=True,
)
def test_from_raw_data(calibration_curve_raw_data):
    y_test, probabilities, clf_names = calibration_curve_raw_data
    plot.CalibrationCurve.from_raw_data(y_test, probabilities, label=clf_names)


@image_comparison(
    baseline_images=["add"],
    extensions=["png"],
    remove_text=True,
)
def test_add(calibration_curve_plot_metrics_data):
    (
        mean_predicted_value_one,
        fraction_of_positives_one,
        clf_names_one,
        mean_predicted_value_two,
        fraction_of_positives_two,
        clf_names_two,
    ) = calibration_curve_plot_metrics_data

    cc1 = plot.CalibrationCurve(
        mean_predicted_value_one, fraction_of_positives_one, clf_names_one
    )
    cc2 = plot.CalibrationCurve(
        mean_predicted_value_two, fraction_of_positives_two, clf_names_two
    )
    cc1 + cc2
