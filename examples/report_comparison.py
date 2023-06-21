import pandas as pd
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn_evaluation.report import compare_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

url = (
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
)

urllib.request.urlretrieve(
    url,
    filename="heart.csv",
)

data = pd.read_csv("heart.csv")

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
