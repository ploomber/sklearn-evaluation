import pytest
from sklearn_evaluation.report import ModelsComparer, compare_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def _get_split_data(data, column="target"):
    X = data.drop(column, axis=1)
    y = data[column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2023
    )

    return X_train, X_test, y_train, y_test


def test_compare_models(heart_dataset, tmp_directory):
    X_train, X_test, y_train, y_test = _get_split_data(heart_dataset)

    model_a = RandomForestClassifier()
    model_a.fit(X_train, y_train)

    model_b = DecisionTreeClassifier()
    model_b.fit(X_train, y_train)

    report = compare_models(model_a, model_b, X_test, y_test)
    report.save("example-compare-report.html")


@pytest.mark.parametrize(
    "model_a, model_b, expected_guidelines, number_of_expected_plots",
    [
        [RandomForestClassifier(), DecisionTreeClassifier(), ["Combined PR"], 3],
        [LogisticRegression(), RandomForestClassifier(), ["Combined PR"], 3],
        [LogisticRegression(), DecisionTreeClassifier(), ["Combined PR"], 3],
        [
            LinearRegression(),
            DecisionTreeClassifier(),
            ["Failed to calculate precision_recall for LinearRegression"],
            1,
        ],
        [
            LinearRegression(),
            LinearRegression(),
            ["Failed to calculate precision_recall for LinearRegression"],
            0,
        ],
        [
            RandomForestClassifier(),
            LinearRegression(),
            ["Failed to calculate precision_recall for LinearRegression"],
            1,
        ],
    ],
)
def test_precision_and_recall(
    heart_dataset, model_a, model_b, expected_guidelines, number_of_expected_plots
):
    X_train, X_test, y_train, y_test = _get_split_data(heart_dataset)

    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    mc = ModelsComparer(model_a, model_b)
    mc.precision_and_recall(X_test, y_test)
    _test_model_results(
        mc, "precision_recall", expected_guidelines, number_of_expected_plots
    )


@pytest.mark.parametrize(
    "model_a, model_b, expected_guidelines, number_of_expected_plots",
    [
        [
            RandomForestClassifier(),
            DecisionTreeClassifier(),
            ["RandomForestClassifier AUC (ROC)", "DecisionTreeClassifier AUC (ROC)"],
            0,
        ],
        [
            LogisticRegression(),
            RandomForestClassifier(),
            ["RandomForestClassifier AUC (ROC)"],
            0,
        ],
        [
            LogisticRegression(),
            DecisionTreeClassifier(),
            ["DecisionTreeClassifier AUC (ROC)"],
            0,
        ],
        [
            LinearRegression(),
            DecisionTreeClassifier(),
            ["DecisionTreeClassifier AUC (ROC)"],
            0,
        ],
        [
            LinearRegression(),
            LinearRegression(),
            [
                "Failed to calculate auc for LinearRegression",
                "Failed to calculate auc for LinearRegression",
            ],
            0,
        ],
        [
            RandomForestClassifier(),
            LinearRegression(),
            [
                "RandomForestClassifier AUC (ROC)",
                "Failed to calculate auc for LinearRegression",
            ],
            0,
        ],
    ],
)
def test_auc(
    heart_dataset, model_a, model_b, expected_guidelines, number_of_expected_plots
):
    X_train, X_test, y_train, y_test = _get_split_data(heart_dataset)

    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    mc = ModelsComparer(model_a, model_b)
    mc.auc(X_test, y_test)
    _test_model_results(mc, "auc", expected_guidelines, number_of_expected_plots)


@pytest.mark.parametrize(
    "model_a, model_b, expected_guidelines, number_of_expected_plots",
    [
        [
            RandomForestClassifier(),
            DecisionTreeClassifier(),
            [
                "RandomForestClassifier compute time is",
                "DecisionTreeClassifier compute time is",
            ],
            0,
        ],
        [
            LogisticRegression(),
            RandomForestClassifier(),
            [
                "LogisticRegression compute time is",
                "RandomForestClassifier compute time is",
            ],
            0,
        ],
        [
            LogisticRegression(),
            DecisionTreeClassifier(),
            [
                "LogisticRegression compute time is",
                "DecisionTreeClassifier compute time is",
            ],
            0,
        ],
        [
            LinearRegression(),
            DecisionTreeClassifier(),
            [
                "LinearRegression compute time is",
                "DecisionTreeClassifier compute time is",
            ],
            0,
        ],
        [
            LinearRegression(),
            LinearRegression(),
            ["LinearRegression compute time is", "LinearRegression compute time is"],
            0,
        ],
        [
            RandomForestClassifier(),
            LinearRegression(),
            [
                "RandomForestClassifier compute time is",
                "LinearRegression compute time is",
            ],
            0,
        ],
    ],
)
def test_computation(
    heart_dataset, model_a, model_b, expected_guidelines, number_of_expected_plots
):
    X_train, X_test, y_train, _ = _get_split_data(heart_dataset)

    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    mc = ModelsComparer(model_a, model_b)
    mc.computation(X_test)
    _test_model_results(
        mc, "prediction_time", expected_guidelines, number_of_expected_plots
    )


@pytest.mark.parametrize(
    "model_a, model_b, expected_guidelines, number_of_expected_plots",
    [
        [RandomForestClassifier(), DecisionTreeClassifier(), [], 2],
        [LogisticRegression(), RandomForestClassifier(), [], 2],
        [LogisticRegression(), DecisionTreeClassifier(), [], 2],
        [
            LinearRegression(),
            DecisionTreeClassifier(),
            ["Failed to calculate calibration for LinearRegression"],
            1,
        ],
        [
            LinearRegression(),
            LinearRegression(),
            [
                "Failed to calculate calibration for LinearRegression",
                "Failed to calculate calibration for LinearRegression",
            ],
            0,
        ],
        [
            RandomForestClassifier(),
            LinearRegression(),
            ["Failed to calculate calibration for LinearRegression"],
            1,
        ],
    ],
)
def test_calibration(
    heart_dataset, model_a, model_b, expected_guidelines, number_of_expected_plots
):
    X_train, X_test, y_train, y_test = _get_split_data(heart_dataset)

    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    mc = ModelsComparer(model_a, model_b)
    mc.calibration(X_test, y_test)
    _test_model_results(
        mc, "calibration", expected_guidelines, number_of_expected_plots
    )


@pytest.mark.parametrize(
    "model_a, model_b, expected_guidelines, number_of_expected_plots",
    [
        [RandomForestClassifier(), DecisionTreeClassifier(), [], 1],
        [LogisticRegression(), RandomForestClassifier(), [], 1],
        [LogisticRegression(), DecisionTreeClassifier(), [], 1],
        [
            LinearRegression(),
            DecisionTreeClassifier(),
            ["Failed to calculate combined_confusion_matrix"],
            0,
        ],
        [
            LinearRegression(),
            LinearRegression(),
            ["Failed to calculate combined_confusion_matrix"],
            0,
        ],
        [
            RandomForestClassifier(),
            LinearRegression(),
            ["Failed to calculate combined_confusion_matrix"],
            0,
        ],
    ],
)
def test_combined_cm(
    heart_dataset, model_a, model_b, expected_guidelines, number_of_expected_plots
):
    X_train, X_test, y_train, y_test = _get_split_data(heart_dataset)

    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    mc = ModelsComparer(model_a, model_b)
    mc.add_combined_cm(X_test, y_test)
    _test_model_results(
        mc, "combined_confusion_matrix", expected_guidelines, number_of_expected_plots
    )


@pytest.mark.parametrize(
    "model_a, model_b, expected_guidelines, number_of_expected_plots",
    [
        [RandomForestClassifier(), DecisionTreeClassifier(), [], 1],
        [LogisticRegression(), RandomForestClassifier(), [], 1],
        [LogisticRegression(), DecisionTreeClassifier(), [], 1],
        [
            LinearRegression(),
            DecisionTreeClassifier(),
            ["Failed to calculate precision_recall"],
            0,
        ],
        [
            LinearRegression(),
            LinearRegression(),
            ["Failed to calculate precision_recall"],
            0,
        ],
        [
            RandomForestClassifier(),
            LinearRegression(),
            ["Failed to calculate precision_recall"],
            0,
        ],
    ],
)
def test_combined_pr(
    heart_dataset, model_a, model_b, expected_guidelines, number_of_expected_plots
):
    X_train, X_test, y_train, y_test = _get_split_data(heart_dataset)

    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    mc = ModelsComparer(model_a, model_b)
    mc.add_combined_pr(X_test, y_test)
    _test_model_results(
        mc, "combined_pr", expected_guidelines, number_of_expected_plots
    )


def _test_model_results(mc, key, expected_guidelines, number_of_expected_plots):
    assert key in mc.evaluation_state
    guidelines = mc.evaluation_state[key]["guidelines"]

    all_plots = [p for p in guidelines if p.__class__.__name__ == "Axes"]
    assert len(all_plots) == number_of_expected_plots

    guidelines = set(guidelines) - set(all_plots)

    for expected_guideline in expected_guidelines:
        assert any(expected_guideline in guideline for guideline in guidelines)


def test_functions_with_none_inputs():
    model_a = RandomForestClassifier()
    model_b = DecisionTreeClassifier()

    me = ModelsComparer(model_a, model_b)

    me.precision_and_recall(None, None)
    me.auc(None, None)
    me.computation(None)
    me.calibration(None, None)
    me.add_combined_cm(None, None)
    me.add_combined_pr(None, None)

    assert len(me.evaluation_state.keys()) == 0


@pytest.mark.parametrize(
    "model_a, model_b",
    [
        [RandomForestClassifier(), None],
        [DecisionTreeClassifier(), None],
        [LogisticRegression(), None],
        [LinearRegression(), None],
        [None, None],
    ],
)
def test_compare_none_models_error(heart_dataset, model_a, model_b):
    _, X_test, _, y_test = _get_split_data(heart_dataset)
    with pytest.raises(ValueError) as e:
        compare_models(model_a, model_b, X_test, y_test)

    assert "Model is none" in str(e.value)
