"""
Tests for precision recall plot.
"""
import pytest
import numpy as np

from matplotlib.testing.decorators import image_comparison

from sklearn_evaluation import plot

# Metrics for binary classification

precision = [
    0.6,
    0.64285714,
    0.69230769,
    0.66666667,
    0.72727273,
    0.7,
    0.77777778,
    0.875,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]
recall = [
    1.0,
    1.0,
    1.0,
    0.88888889,
    0.88888889,
    0.77777778,
    0.77777778,
    0.77777778,
    0.55555556,
    0.44444444,
    0.33333333,
    0.22222222,
    0.11111111,
    0.0,
]


# Metrics for multiclass classification
recall_multi = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6666666666666666, 0.3333333333333333, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 0.0],
]
precision_multi = [
    [
        0.3333333333333333,
        0.375,
        0.42857142857142855,
        0.5,
        0.6,
        0.75,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    [0.3333333333333333, 0.5, 1.0, 1.0],
    [0.3333333333333333, 0.5, 1.0, 1.0],
    [0.3333333333333333, 0.5, 1.0, 1.0],
]


@pytest.mark.parametrize(
    "recall, precision",
    [
        (recall, precision),
        (np.array(recall), np.array(precision)),
        ([recall], [precision]),
    ],
)
@image_comparison(
    baseline_images=["precision_recall"], extensions=["png"], remove_text=True
)
def test_precision_recall(precision, recall):
    plot.PrecisionRecall(precision, recall).plot()


@pytest.mark.parametrize(
    "recall, precision",
    [
        (recall, precision),
        (np.array(recall), np.array(precision)),
        ([recall], [precision]),
    ],
)
@image_comparison(
    baseline_images=["precision_recall_with_label"],
    extensions=["png"],
    remove_text=True,
)
def test_precision_recall_with_label(precision, recall):
    plot.PrecisionRecall(precision, recall, label="Test With Label").plot()


@image_comparison(
    baseline_images=["precision_recall_multi"], extensions=["png"], remove_text=True
)
def test_precision_recall_multi():
    plot.PrecisionRecall(precision_multi, recall_multi).plot()


@image_comparison(
    baseline_images=["precision_recall_multi_with_label"],
    extensions=["png"],
    remove_text=True,
)
def test_precision_recall_multi_with_label():
    plot.PrecisionRecall(
        precision_multi,
        recall_multi,
        label=["Macro-average", "Setosa", "Versicolor", "Virginica"],
    ).plot()


def test_metrics_length_mismatch(ploomber_value_error_message):
    with pytest.raises(ValueError) as e:
        plot.PrecisionRecall([2.9, 1.2], [6.0]).plot()
    assert "precision and recall lengths should correspond" in str(e.value)


@image_comparison(
    baseline_images=["from_raw_data_binary"],
    extensions=["png"],
    remove_text=True,
    tol=1.0,
)
def test_from_raw_data_binary(precision_recall_binary_classification):
    y_true, y_score = precision_recall_binary_classification
    plot.PrecisionRecall.from_raw_data(y_true, y_score)


@image_comparison(
    baseline_images=["from_raw_data_binary_with_label"],
    extensions=["png"],
    remove_text=True,
    tol=1.0,
)
def test_from_raw_data_binary_with_label(precision_recall_binary_classification):
    y_true, y_score = precision_recall_binary_classification
    plot.PrecisionRecall.from_raw_data(y_true, y_score, label="Test Label")


@image_comparison(
    baseline_images=["from_raw_data_multi"],
    extensions=["png"],
    remove_text=True,
    tol=2.0,
)
def test_from_raw_data_multiclass(
    precision_recall_multiclass_classification_set_one,
):
    y_true, y_score = precision_recall_multiclass_classification_set_one
    plot.PrecisionRecall.from_raw_data(y_true, y_score)


@image_comparison(
    baseline_images=["from_raw_data_multi_label"],
    extensions=["png"],
    remove_text=True,
    tol=2.0,
)
def test_from_raw_data_multiclass_with_label(
    precision_recall_multiclass_classification_set_one,
):
    y_true, y_score = precision_recall_multiclass_classification_set_one
    plot.PrecisionRecall.from_raw_data(
        y_true, y_score, label=["Setosa", "Versicolor", "Virginica"]
    )


def test_add(
    precision_recall_multiclass_classification_set_one,
    precision_recall_multiclass_classification_set_two,
):
    y_true_one, y_score_one = precision_recall_multiclass_classification_set_one
    pr1 = plot.PrecisionRecall.from_raw_data(y_true_one, y_score_one)
    y_true_two, y_score_two = precision_recall_multiclass_classification_set_two
    pr2 = plot.PrecisionRecall.from_raw_data(y_true_two, y_score_two)
    pr1 + pr2
