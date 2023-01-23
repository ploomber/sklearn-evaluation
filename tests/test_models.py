from sklearn_evaluation.models import evaluate_model


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def test_evaluate_model(capsys):

    import urllib.request
    import pandas as pd

    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/sharmaroshan/"
        "Heart-UCI-Dataset/master/heart.csv",
        filename="heart.csv",
    )

    data = pd.read_csv("heart.csv")

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
    # report.save("example-report.html")

    import pathlib
    file = pathlib.Path("heart.csv")
    file.unlink()
    # report.save("example-report.html")
