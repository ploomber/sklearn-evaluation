import pytest
import numpy as np
from sklearn_evaluation import plot, __version__


@pytest.fixture
def y():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    return y_true, y_pred


@pytest.fixture
def X():
    X_test = [
        [0.8, -0.754, 0.93, -1.224, 0.793],
        [-0.089, 0.334, -0.178, 0.402, -0.851],
        [0.568, -0.802, -0.005, -0.741, 1.103],
        [-0.731, 1.169, -0.397, 1.3, -1.97],
        [1.94, -1.702, 0.868, -2.089, 1.035],
        [0.886, -1.451, -2.138, -0.162, 1.529],
        [0.228, -0.094, 1.87, -1.125, 0.458],
        [-0.425, -0.399, -0.844, 0.125, 1.807],
    ]
    return X_test


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
    cm = plot.InteractiveConfusionMatrix.from_raw_data(
        y_true, y_pred, normalize=normalize, target_names=target_names
    )

    denominator = 4 if normalize else 1
    value = 2 / denominator

    assert cm._get_data() == {
        "class": "sklearn_evaluation.plot.InteractiveConfusionMatrix",
        "cm": [[value, value], [value, value]],
        "target_names": target_names_expected,
        "interactive_data": None,
        "version": __version__,
    }


def test_from_raw_data_with_x_pandas(X, y):
    import pandas as pd

    y_true, y_pred = y
    X_test = X
    columns = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
    df = pd.DataFrame(X_test, columns=columns)
    cm = plot.InteractiveConfusionMatrix.from_raw_data(y_true, y_pred, X_test=df)

    interactive_data_keys_expected = [
        "Feature A sampled",
        "Feature B sampled",
        "Feature C sampled",
        "Feature D sampled",
        "Feature A metric",
        "Feature B metric",
        "Feature C metric",
        "Feature D metric",
        "Feature E metric",
    ]

    assert [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ] in cm._get_data()["interactive_data"]["quadrant statistics name"]
    for key in interactive_data_keys_expected:
        assert key in list(cm._get_data()["interactive_data"].keys())


@pytest.mark.parametrize(
    "feature_names, interactive_data_keys_expected",
    [
        (
            None,
            [
                "Feature 0 sampled",
                "Feature 1 sampled",
                "Feature 2 sampled",
                "Feature 3 sampled",
                "Feature 4 sampled",
                "Feature 0 metric",
                "Feature 1 metric",
                "Feature 2 metric",
                "Feature 2 metric",
                "Feature 4 metric",
            ],
        ),
        (
            ["A", "B", "C", "D", "E"],
            [
                "A sampled",
                "B sampled",
                "C sampled",
                "D sampled",
                "E sampled",
                "A metric",
                "B metric",
                "C metric",
                "D metric",
                "E metric",
            ],
        ),
    ],
)
def test_from_raw_data_with_x_numpy(
    feature_names, interactive_data_keys_expected, X, y
):
    y_true, y_pred = y
    X_test = X
    cm = plot.InteractiveConfusionMatrix.from_raw_data(
        y_true,
        y_pred,
        feature_names=feature_names,
        X_test=X_test,
    )

    for key in interactive_data_keys_expected:
        assert key in list(cm._get_data()["interactive_data"].keys())


@pytest.mark.parametrize("bad_y_true_value", [None, [], [1, 2, 3, 4, 5], [1, 0], ""])
def test_confusion_matrix_bad_input_value_error(
    y, bad_y_true_value, ploomber_value_error_message
):
    _, y_pred = y
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.InteractiveConfusionMatrix.from_raw_data(bad_y_true_value, y_pred)


def test_from_raw_data_mixed_datatype(X, y):
    import pandas as pd

    y_true, y_pred = y
    X_test = [
        [0.8, -0.754, 0.93, -1.224, 0.793],
        ["a", "x", "a", "y", "a"],
        [0.568, -0.802, -0.005, -0.741, 1.103],
        [-0.731, 1.169, -0.397, 1.3, -1.97],
        [1.94, -1.702, 0.868, -2.089, 1.035],
        [1.23, -1.452, 0.848, -2.081, 2.045],
        [0.228, -0.094, 1.87, -1.125, 0.458],
        [-0.425, -0.399, -0.844, 0.125, 1.807],
    ]
    columns = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
    df = pd.DataFrame(X_test, columns=columns)
    cm = plot.InteractiveConfusionMatrix.from_raw_data(y_true, y_pred, X_test=df)

    interactive_data_keys_expected = [
        "Feature A sampled",
        "Feature B sampled",
        "Feature C sampled",
        "Feature D sampled",
        "Feature A metric",
        "Feature B metric",
        "Feature C metric",
        "Feature D metric",
        "Feature E metric",
    ]

    for key in interactive_data_keys_expected:
        assert key in list(cm._get_data()["interactive_data"].keys())


def test_confusion_matrix_bad_feature_names_error(y, X, ploomber_value_error_message):
    y_true, y_pred = y
    X_test = X
    feature_names = ["Feature 1", "Feature 2"]
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.InteractiveConfusionMatrix.from_raw_data(
            y_true, y_pred, X_test=X_test, feature_names=feature_names
        )


def test_confusion_matrix_nan_error(y, X, ploomber_value_error_message):
    y_true, y_pred = y
    X_test = [
        [0.8, -0.754, 0.93, -1.224, 0.793],
        [0.568, -0.802, float("nan"), -0.741, 1.103],
        [0.568, -0.802, -0.005, -0.741, 1.103],
        [-0.731, 1.169, -0.397, 1.3, -1.97],
        [1.94, -1.702, float("nan"), float("nan"), 1.035],
        [1.23, -1.452, 0.848, -2.081, 2.045],
        [0.228, -0.094, 1.87, -1.125, 0.458],
        [-0.425, -0.399, -0.844, 0.125, 1.807],
    ]
    with pytest.raises(ValueError, match=ploomber_value_error_message):
        plot.InteractiveConfusionMatrix.from_raw_data(y_true, y_pred, X_test=X_test)
