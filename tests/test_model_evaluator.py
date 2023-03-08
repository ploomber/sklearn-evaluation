import pytest
from sklearn_evaluation.report import ModelEvaluator, evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def _get_classifier_model_values(model, data, column="fbs"):
    X = data.drop(column, axis=1)
    y = data[column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2023
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if isinstance(model, LinearRegression):
        y_score = None
    else:
        y_score = model.predict_proba(X_test)

    return y_test, y_pred, y_score


@pytest.mark.parametrize(
    "model, skip_y_pred",
    [
        [RandomForestClassifier(), False],
        [DecisionTreeClassifier(), False],
        [LogisticRegression(), False],
        [LinearRegression(), True],
    ],
)
def test_evaluate_model(heart_dataset, model, skip_y_pred, tmp_directory):
    data = heart_dataset

    column = "fbs"
    X = data.drop(column, axis=1)
    y = data[column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2023
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_score = None if skip_y_pred else model.predict_proba(X_test)

    if skip_y_pred:
        y_test = None

    report = evaluate_model(model, y_test, y_pred, y_score=y_score, X_test=X_test)
    report.save("example-report.html")


def test_model_evaluator_functions_with_none_inputs():
    model = RandomForestClassifier()
    me = ModelEvaluator(model)

    me.evaluate_balance(None)
    me.evaluate_accuracy(None, None)
    me.evaluate_auc(None)
    me.generate_general_stats(None, None, None)
    me.evaluate_precision_and_recall(None, None)
    me.evaluate_auc(None, None)
    me.get_model_prediction_time(None)
    me.evaluate_calibration(None, None)

    assert len(me.evaluation_state.keys()) == 0


@pytest.mark.parametrize(
    "model, expected_name",
    [
        [RandomForestClassifier(), "RandomForestClassifier"],
        [DecisionTreeClassifier(), "DecisionTreeClassifier"],
        [LogisticRegression(), "LogisticRegression"],
        [LinearRegression(), "LinearRegression"],
    ],
)
def test_get_model_name(model, expected_name):
    me = ModelEvaluator(model)
    assert me._get_model_name(model) == expected_name


def test_model_evaluator_evaluate_model_with_none_inputs(heart_dataset):
    model = RandomForestClassifier()
    data = heart_dataset
    _, _, _ = _get_classifier_model_values(model, data)

    me = ModelEvaluator(model)
    evaluate_model(model, None, None, None)

    assert len(me.evaluation_state.keys()) == 0


def test_evaluate_model_none_error(heart_dataset):
    model = RandomForestClassifier()
    data = heart_dataset
    y_test, y_pred, y_score = _get_classifier_model_values(model, data)

    with pytest.raises(ValueError) as e:
        evaluate_model(None, y_test, y_pred, y_score)

    assert "Model is none" in str(e.value)


@pytest.mark.parametrize(
    "model, expected_guidelines, number_of_expected_plots, data_to_split",
    [
        [RandomForestClassifier(), ["imbalanced"], True, "fbs"],
        [RandomForestClassifier(), ["Your dataset is balanced"], False, "target"],
        [DecisionTreeClassifier(), ["imbalanced"], True, "fbs"],
        [DecisionTreeClassifier(), ["Your dataset is balanced"], False, "target"],
        [LogisticRegression(), ["imbalanced"], True, "fbs"],
        [LogisticRegression(), ["Your dataset is balanced"], False, "target"],
        [LinearRegression(), ["imbalanced"], True, "fbs"],
        [LinearRegression(), ["Your dataset is balanced"], False, "target"],
    ],
)
def test_model_evaluator_evaluate_balance(
    heart_dataset, model, expected_guidelines, number_of_expected_plots, data_to_split
):
    y_test, _, _ = _get_classifier_model_values(model, heart_dataset, data_to_split)

    me = ModelEvaluator(model)
    me.evaluate_balance(y_test)

    _test_model_results(me, "balance", expected_guidelines, number_of_expected_plots)


@pytest.mark.parametrize(
    "model, expected_guidelines, number_of_expected_plots",
    [
        [RandomForestClassifier(), ["Accuracy is"], 0],
        [DecisionTreeClassifier(), ["Accuracy is"], 0],
        [LogisticRegression(), ["Accuracy is"], 0],
    ],
)
def test_model_evaluator_accuracy(
    heart_dataset, model, expected_guidelines, number_of_expected_plots
):
    y_test, y_pred, _ = _get_classifier_model_values(model, heart_dataset)

    me = ModelEvaluator(model)
    me.evaluate_accuracy(y_test, y_pred)

    _test_model_results(me, "accuracy", expected_guidelines, number_of_expected_plots)


@pytest.mark.parametrize(
    "model, expected_guidelines, number_of_expected_plots",
    [
        [RandomForestClassifier(), ["Area under curve is"], 1],
        [DecisionTreeClassifier(), ["Area under curve is"], 1],
        [LogisticRegression(), ["Area under curve is"], 1],
    ],
)
def test_model_evaluator_auc(
    heart_dataset, model, expected_guidelines, number_of_expected_plots
):
    y_test, _, y_score = _get_classifier_model_values(model, heart_dataset)

    me = ModelEvaluator(model)
    me.evaluate_auc(y_test, y_score)

    _test_model_results(me, "auc", expected_guidelines, number_of_expected_plots)


@pytest.mark.parametrize(
    "model, expected_guidelines, number_of_expected_plots",
    [
        [RandomForestClassifier(), [], 2],
        [DecisionTreeClassifier(), [], 2],
        [LogisticRegression(), [], 2],
    ],
)
def test_model_evaluator_generate_general_stats(
    heart_dataset, model, expected_guidelines, number_of_expected_plots
):
    y_test, y_pred, y_score = _get_classifier_model_values(model, heart_dataset)

    me = ModelEvaluator(model)
    me.generate_general_stats(y_test, y_pred, y_score)

    _test_model_results(
        me, "general_stats", expected_guidelines, number_of_expected_plots
    )


def _test_model_results(model, key, expected_guidelines, number_of_expected_plots):
    assert key in model.evaluation_state
    guidelines = model.evaluation_state[key]["guidelines"]

    all_plots = [p for p in guidelines if p.__class__.__name__ == "Axes"]
    assert len(all_plots) == number_of_expected_plots

    guidelines = set(guidelines) - set(all_plots)

    for expected_guideline in expected_guidelines:
        assert any(expected_guideline in str(guideline) for guideline in guidelines)
