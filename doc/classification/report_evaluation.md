---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
myst:
  html_meta:
    description lang=en: "Evaluate ML models using the heart disease dataset and sklearn-evaluation. Learn model preparation, training, prediction, and evaluation."
    keywords: "evaluation, heart disease dataset, machine learning, model preparation, model training, model prediction, performance metrics, sklearn-evaluation, RandomForestClassifier, report generation, model performance"
    property=og:locale: "en_US"
notebook_metadata_filter: myst
---

# Report: Evaluation

```{versionadded} 0.11.4
```

We use different metrics to estimate a machine learning model's performance, and to understand its strengths and weaknesses.

In this guide, we'll show you how to easily generate a report with everything your need in one place using our [`evaluate_models`](../api/plot.rst).

We'll use the heart disease dataset, you can download it from <a href="https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data" target="_blank">here</a>.

## Download the data

```{code-cell} ipython3
import urllib.request
import pandas as pd

url = (
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
)

urllib.request.urlretrieve(
    url,
    filename="heart.csv",
)

data = pd.read_csv("heart.csv")
```

## Prepare the data

```{code-cell} ipython3
from sklearn.model_selection import train_test_split

column = "fbs"
X = data.drop(column, axis=1)
y = data[column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2023
)
```

## Define the model

```{code-cell} ipython3
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)
```

## Evaluate the model

```{code-cell} ipython3
:tags: [remove-output]

from sklearn_evaluation.report import evaluate_model

report = evaluate_model(model, y_test, y_pred, y_score=y_score)
```

## Embed the report

```{code-cell} ipython3
report
```

## Save report as HTML

```{code-cell} ipython3
report.save("report.html")
```