from sklearn.ensemble import RandomForestClassifier
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn_evaluation.models import evaluate_model

urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/sharmaroshan/' +
    'Heart-UCI-Dataset/master/heart.csv', filename='heart.csv')

data = pd.read_csv('heart.csv')


column = 'fbs'
X = data.drop(column, axis=1)
y = data[column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2023)


model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)

report = evaluate_model(model, y_test, y_pred, y_score=y_score)
