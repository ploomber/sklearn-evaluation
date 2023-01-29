import pytest
from sklearn_evaluation.models import evaluate_model, compare_models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pathlib


@pytest.fixture
def heart_dataset():
    import urllib.request
    import pandas as pd

    file_name = "heart.csv"

    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/sharmaroshan/"
        "Heart-UCI-Dataset/master/heart.csv",
        filename=file_name,
    )

    data = pd.read_csv(file_name)

    return data, file_name


def test_evaluate_model(heart_dataset):
    data, file_name = heart_dataset

    column = "fbs"
    X = data.drop(column, axis=1)
    y = data[column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2023
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    report = evaluate_model(y_test, y_pred, model=model, y_score=y_score)  # noqa
    report.save("example-report.html")

    _clean_files([file_name])


def test_compare_models(heart_dataset):
    data, file_name = heart_dataset

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

    report = compare_models(model_a, model_b, X_train, X_test, y_test)  # noqa
    report.save("example-compare-report.html")

    _clean_files([file_name])


def _clean_files(files):
    for file in files:
        pathlib.Path(file).unlink()


def test_ido():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn import tree
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    from sklearn_evaluation import plot, table
    # Based on https://github.com/Adeyinka-hub/Machine-Learning-2/blob/master/Penguin%20Dataset.ipynb

    df = sns.load_dataset('penguins')

    df.isnull().sum()
    df.dropna(inplace=True)
    Y = df.species
    Y = Y.map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
    df.drop('species', inplace=True, axis=1)
    se = pd.get_dummies(df['sex'], drop_first=True)
    df = pd.concat([df, se], axis=1)
    df.drop('sex', axis=1, inplace=True)
    le = LabelEncoder()
    df['island'] = le.fit_transform(df['island'])

    X = df
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=40)

    model = RandomForestClassifier()
    dt_model = model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    print("Acc on test data: {:,.3f}".format(model.score(X_test, y_test)))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    report = evaluate_model(y_test, y_pred, model=model, y_score=y_score)  # noqa
    report.save("example-report.html")
