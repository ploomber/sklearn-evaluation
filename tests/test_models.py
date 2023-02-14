import pytest
from sklearn_evaluation.models import evaluate_model, compare_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import urllib.request
import pandas as pd
from sklearn_evaluation.models import ModelEvaluator


@pytest.fixture
def heart_dataset(tmp_directory):
    file_name = "heart.csv"

    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/sharmaroshan/"
        "Heart-UCI-Dataset/master/heart.csv",
        filename=file_name,
    )

    data = pd.read_csv(file_name)

    return data


def _get_classifier_model_values(model, data, column="fbs"):
    X = data.drop(column, axis=1)
    y = data[column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2023
    )
    model = model
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

    report = evaluate_model(model, y_test, y_pred, y_score=y_score)
    report.save("example-report.html")


def test_compare_models(heart_dataset, tmp_directory):
    data = heart_dataset

    column = "target"
    X = data.drop(column, axis=1)
    y = data[column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2023
    )

    model_a = RandomForestClassifier()
    model_a.fit(X_train, y_train)

    model_b = DecisionTreeClassifier()
    model_b.fit(X_train, y_train)

    report = compare_models(model_a, model_b, X_test, y_test)
    report.save("example-compare-report.html")


def test_model_evaluator_functions_with_none_inputs(heart_dataset):
    model = RandomForestClassifier()
    data = heart_dataset
    _, _, _ = _get_classifier_model_values(model, data)

    me = ModelEvaluator(model)

    me.evaluate_balance(None)
    me.evaluate_accuracy(None, None)
    me.evaluate_auc(None)
    me.generate_general_stats(None)

    assert len(me.evaluation_state.keys()) == 0


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
    "model, column, expected_guide_line, test_plot",
    [
        [RandomForestClassifier(), "fbs", "imbalanced", True],
        [RandomForestClassifier(), "target", "your model is balanced", False],
        [DecisionTreeClassifier(), "fbs", "imbalanced", True],
        [DecisionTreeClassifier(), "target", "your model is balanced", False],
        [LogisticRegression(), "fbs", "imbalanced", True],
        [LogisticRegression(), "target", "your model is balanced", False],
        [LinearRegression(), "fbs", "imbalanced", True],
        [LinearRegression(), "target", "your model is balanced", False],
    ],
)
def test_model_evaluator_evaluate_balance(
    heart_dataset, model, column, expected_guide_line, test_plot
):
    data = heart_dataset
    y_test, _, _ = _get_classifier_model_values(model, data, column)

    me = ModelEvaluator(model)
    me.evaluate_balance(y_test)

    key = "balance"
    assert key in me.evaluation_state
    guidelines = me.evaluation_state[key]["guidelines"]
    assert expected_guide_line in guidelines[0]

    if test_plot:
        assert guidelines[1].__class__.__name__ == "AxesSubplot"


@pytest.mark.parametrize(
    "model, column, expected_guide_line, test_plot",
    [
        [RandomForestClassifier(), "fbs", "Accuracy is", False],
        [DecisionTreeClassifier(), "fbs", "Accuracy is", False],
        [LogisticRegression(), "fbs", "Accuracy is", False],
    ],
)
def test_model_evaluator_accuracy(
    heart_dataset, model, column, expected_guide_line, test_plot
):
    data = heart_dataset
    y_test, y_pred, _ = _get_classifier_model_values(model, data, column)

    me = ModelEvaluator(model)
    me.evaluate_accuracy(y_test, y_pred)

    key = "accuracy"
    assert key in me.evaluation_state
    guidelines = me.evaluation_state[key]["guidelines"]
    assert expected_guide_line in guidelines[0]

    if test_plot:
        assert guidelines[1].__class__.__name__ == "AxesSubplot"


@pytest.mark.parametrize(
    "model, column, expected_guide_line, test_plot",
    [
        [RandomForestClassifier(), "fbs", "Area under curve is", True],
        [DecisionTreeClassifier(), "fbs", "Area under curve is", True],
        [LogisticRegression(), "fbs", "Area under curve is", True],
    ],
)
def test_model_evaluator_auc(
    heart_dataset, model, column, expected_guide_line, test_plot
):
    data = heart_dataset
    y_test, _, y_score = _get_classifier_model_values(model, data, column)

    me = ModelEvaluator(model)
    me.evaluate_auc(y_test, y_score)

    key = "auc"
    assert key in me.evaluation_state
    guidelines = me.evaluation_state[key]["guidelines"]
    assert expected_guide_line in guidelines[0]

    if test_plot:
        assert guidelines[1].__class__.__name__ == "AxesSubplot"


@pytest.mark.parametrize(
    "model",
    [
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        LogisticRegression(),
    ],
)
def test_model_evaluator_generate_general_stats(heart_dataset, model):
    data = heart_dataset
    y_test, y_pred, y_score = _get_classifier_model_values(model, data)

    me = ModelEvaluator(model)
    me.generate_general_stats(y_test, y_pred, y_score)

    key = "general_stats"
    assert key in me.evaluation_state
    guidelines = me.evaluation_state[key]["guidelines"]

    assert guidelines[0].__class__.__name__ == "AxesSubplot"
    assert guidelines[1].__class__.__name__ == "AxesSubplot"
