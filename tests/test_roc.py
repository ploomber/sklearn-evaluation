import pytest
import numpy as np
from sklearn_evaluation import plot, __version__

from functools import partial
import sys
from matplotlib.testing.decorators import image_comparison as _image_comparison
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# older versions of Python are not compatible with the latest version of
# matplotlib, which leads to image differences. We increase the tolerance in
# such cases
image_comparison = partial(
    _image_comparison,
    tol=1.7 if sys.version_info.minor in (6, 7) else 0,
    remove_text=True,
    extensions=["png"],
)

fpr = [0.0, 0.2, 0.4, 0.4, 0.6, 1.0]
tpr = [0.0, 0.2, 0.4, 1.0, 1.0, 1.0]


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


@pytest.fixture
def roc_multi_classification_values():
    roc_rates_n_classes = [
        {
            "fpr": [
                0.0,
                0.0,
                0.11764705882352941,
                0.11764705882352941,
                0.23529411764705882,
                0.23529411764705882,
                1.0,
            ],
            "tpr": [
                0.0,
                0.3333333333333333,
                0.3333333333333333,
                0.6666666666666666,
                0.6666666666666666,
                1.0,
                1.0,
            ],
        },
        {
            "fpr": [
                0.0,
                0.0,
                0.058823529411764705,
                0.058823529411764705,
                0.17647058823529413,
                0.17647058823529413,
                1.0,
            ],
            "tpr": [
                0.0,
                0.3333333333333333,
                0.3333333333333333,
                0.6666666666666666,
                0.6666666666666666,
                1.0,
                1.0,
            ],
        },
        {
            "fpr": [
                0.0,
                0.0,
                0.0,
                0.26666666666666666,
                0.26666666666666666,
                0.7333333333333333,
                0.7333333333333333,
                1.0,
            ],
            "tpr": [0.0, 0.2, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0],
        },
        {
            "fpr": [0.0, 0.0, 0.0, 0.13333333333333333, 0.13333333333333333, 1.0],
            "tpr": [0.0, 0.2, 0.4, 0.4, 1.0, 1.0],
        },
        {"fpr": [0.0, 0.0, 0.0, 1.0], "tpr": [0.0, 0.25, 1.0, 1.0]},
    ]

    avg_fpr = [
        0.0,
        0.0,
        0.0,
        0.0125,
        0.0125,
        0.0375,
        0.0375,
        0.0625,
        0.0625,
        0.075,
        0.075,
        0.0875,
        0.0875,
        0.125,
        0.125,
        0.15,
        0.15,
        0.2125,
        0.2125,
        0.35,
        0.35,
        0.6,
        0.6,
        1.0,
    ]
    avg_tpr = [
        0.0,
        0.05,
        0.35,
        0.35,
        0.4,
        0.4,
        0.5,
        0.5,
        0.55,
        0.55,
        0.6,
        0.6,
        0.65,
        0.65,
        0.7,
        0.7,
        0.75,
        0.75,
        0.8,
        0.8,
        0.95,
        0.95,
        1.0,
        1.0,
    ]

    fpr = [avg_fpr]
    tpr = [avg_tpr]
    labels = ["micro-average ROC curve"]

    for i in range(len(roc_rates_n_classes)):
        fpr_tpr = roc_rates_n_classes[i]
        fpr.append(fpr_tpr["fpr"])
        tpr.append(fpr_tpr["tpr"])
        labels.append(f"(class {i}) ROC curve")

    return fpr, tpr, labels


@pytest.fixture
def roc_multi_classification_raw_data(target_analysis_multiclass):
    X_train, X_test, y_train, y_test = target_analysis_multiclass
    classifier = LogisticRegression()
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

    return y_test, y_score


@pytest.fixture
def roc_multi_classification_raw_data_set2():
    from sklearn.datasets import load_iris

    iris = load_iris()
    X, y = iris.data, iris.target
    y = iris.target_names[y]

    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape

    X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

    classifier = LogisticRegression()
    y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

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


@pytest.mark.parametrize(
    "fpr, tpr",
    [(fpr, tpr)],
)
def test_roc_sub_not_implemented_error(fpr, tpr):
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


@pytest.mark.parametrize(
    "bad_y_scores", [np.array([0, 1, 2, 0, 1]), np.array([[0, 0, 1], [1, 0, 0]])]
)
def test_plot_roc_bad_y_score_input_value_error(bad_y_scores):
    y_test = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    # test classes example
    with pytest.raises(ValueError, match="Please check y_score values"):
        plot.roc(y_test, bad_y_scores)

    with pytest.raises(ValueError, match="Please check y_score values"):
        plot.ROC.from_raw_data(y_test, bad_y_scores)


@pytest.mark.parametrize(
    "bad_y_true",
    [
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.1, 0.1, 0.3],
            ]
        ),
        np.array(
            [0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.1, 0.1, 0.3],
        ),
    ],
)
def test_plot_roc_bad_y_true_input_value_error(bad_y_true, y):
    _, y_score = y

    # test classes example
    with pytest.raises(ValueError, match="Please check y_true values"):
        plot.roc(bad_y_true, y_score)

    with pytest.raises(ValueError, match="Please check y_true values"):
        plot.ROC.from_raw_data(bad_y_true, y_score)


@pytest.mark.parametrize(
    "y_test, y_score",
    [
        (np.array([0, 1]), np.array([1, 1])),
        (np.array([0, 1, 1]), np.array([[1, 1], [0, 1], [0, 1]])),
    ],
)
@image_comparison(baseline_images=["roc_binary_like_array", "roc_binary_like_array"])
def test_plot_roc_valid_input_binary_like_array(y_test, y_score):
    plot.roc(y_test, y_score)
    plot.ROC.from_raw_data(y_test, y_score)


@image_comparison(baseline_images=["roc_multi"])
def test_roc_from_raw_data_y_test_binary(roc_multi_classification_raw_data):
    y_test, y_score = roc_multi_classification_raw_data

    y_test_bin = np.zeros((y_test.size, y_test.max() + 1))
    y_test_bin[np.arange(y_test.size), y_test] = 1
    plot.ROC.from_raw_data(y_test_bin, y_score)


@image_comparison(baseline_images=["roc", "roc"])
def test_roc_from_raw_data(y):
    y_test, y_score = y
    # roc_old_api
    plot.roc(y_test, y_score)

    plot.ROC.from_raw_data(y_test, y_score)


@image_comparison(baseline_images=["roc", "roc"])
def test_roc_y_score_vector(y):
    y_test, y_score = y
    y_score_vector = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.1, 0.1, 0.3])

    # roc_old_api
    plot.roc(y_test, y_score_vector)

    plot.ROC.from_raw_data(y_test, y_score)


@pytest.mark.parametrize(
    "fpr, tpr",
    [(fpr, tpr)],
)
@image_comparison(baseline_images=["roc", "roc"])
def test_roc(fpr, tpr):
    # test when list is given
    roc = plot.ROC(fpr, tpr)
    roc.plot()

    # test when np.array
    roc = plot.ROC(np.array(fpr), np.array(tpr))
    roc.plot()


@image_comparison(baseline_images=["roc_multi"])
def test_roc_multi_from_raw_data(roc_multi_classification_raw_data):
    y_test, y_score = roc_multi_classification_raw_data
    plot.ROC.from_raw_data(y_test, y_score)


@image_comparison(baseline_images=["roc_multi_with_labels"])
def test_roc_multi_from_raw_data_with_labels(roc_multi_classification_raw_data_set2):
    y_test, y_score = roc_multi_classification_raw_data_set2
    plot.ROC.from_raw_data(y_test, y_score)


@image_comparison(baseline_images=["roc_multi", "roc_multi"])
def test_roc_multi(roc_multi_classification_values):
    fpr, tpr, label = roc_multi_classification_values
    # test when fpr and tpr are lists
    multi_roc = plot.ROC(fpr, tpr, label=label)
    multi_roc.plot()

    # test when fpr and tpr are np.arrays
    fpr_np_array = [np.array(fpr_list) for fpr_list in fpr]
    tpr_np_array = [np.array(tpr_list) for tpr_list in tpr]
    multi_roc = plot.ROC(fpr_np_array, tpr_np_array, label=label)
    multi_roc.plot()


@image_comparison(baseline_images=["roc", "roc2", "roc_add_roc"])
def test_roc_add_to_roc_from_raw_data(y):
    y_test, y_score = y
    y_test_roc2 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_score_roc2 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.3, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.2, 0.3, 0.5],
        ]
    ).T

    roc1 = plot.ROC.from_raw_data(y_test, y_score)
    roc2 = plot.ROC.from_raw_data(y_test_roc2, y_score_roc2)
    roc1 + roc2


@pytest.mark.parametrize(
    "fpr, tpr",
    [(fpr, tpr)],
)
@image_comparison(baseline_images=["roc_add_roc"])
def test_roc_add_to_roc(fpr, tpr):
    fpr2 = [0.0, 0.2, 0.4, 0.4, 0.6, 0.8, 1.0]
    tpr2 = [0.0, 0.2, 0.4, 0.8, 0.8, 1.0, 1.0]

    roc1 = plot.ROC(fpr, tpr)
    roc2 = plot.ROC(fpr2, tpr2)

    roc1 + roc2


@pytest.mark.parametrize(
    "fpr, tpr",
    [(fpr, tpr)],
)
@image_comparison(baseline_images=["roc_add_multi"])
def test_roc_add_to_multi(fpr, tpr, roc_multi_classification_values):
    fpr2, tpr2, label = roc_multi_classification_values

    roc1 = plot.ROC(fpr, tpr)
    roc2 = plot.ROC(fpr2, tpr2, label=label)
    roc1 + roc2


@pytest.mark.parametrize(
    "fpr2, tpr2",
    [(fpr, tpr)],
)
@image_comparison(baseline_images=["multi_add_roc"])
def test_roc_multi_add_to_roc(fpr2, tpr2, roc_multi_classification_values):
    fpr1, tpr1, label = roc_multi_classification_values

    roc1 = plot.ROC(fpr1, tpr1, label=label)
    roc2 = plot.ROC(fpr2, tpr2)
    roc1 + roc2


@image_comparison(
    baseline_images=["roc_multi", "roc_multi_set2", "roc_multi_add_multi"]
)
def test_roc_multi_add_to_multi_from_raw_data(
    roc_multi_classification_raw_data, roc_multi_classification_raw_data_set2
):
    y_test, y_score = roc_multi_classification_raw_data
    y_test_roc2, y_score_roc2 = roc_multi_classification_raw_data_set2

    roc1 = plot.ROC.from_raw_data(y_test, y_score)
    roc2 = plot.ROC.from_raw_data(y_test_roc2, y_score_roc2)
    roc1 + roc2


@image_comparison(
    baseline_images=["roc_multi", "roc_multi_set2", "roc_multi_add_multi"]
)
def test_roc_multi_add_to_multi(
    roc_multi_classification_values, roc_multi_classification_raw_data_set2
):
    fpr1, tpr1, label = roc_multi_classification_values
    y_test_roc2, y_score_roc2 = roc_multi_classification_raw_data_set2

    roc1 = plot.ROC(fpr1, tpr1, label=label).plot()
    roc2 = plot.ROC.from_raw_data(y_test_roc2, y_score_roc2)
    roc1 + roc2
