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
---

# Model comparison

+++

## Prepare heart disease dataset

```{code-cell} ipython3
from sklearn_evaluation.models import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import urllib.request
import pandas as pd
```

```{code-cell} ipython3
# download dataset. Reference: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv', filename='heart.csv')

data = pd.read_csv('heart.csv')
```

## Prepare data 
```{code-cell} ipython3
column = "target"
X = data.drop(column, axis=1)
y = data[column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2023
)

```

## Train model a using RandomForestClassifier
```{code-cell} ipython3
model_a = RandomForestClassifier()
model_a.fit(X_train, y_train)
```

## Train model b using DecisionTreeClassifier
```{code-cell} ipython3
model_b = DecisionTreeClassifier()
model_b.fit(X_train, y_train)
```

## Compare models
```
report = compare_models(model_a, model_b, X_train, X_test, y_test)
```

## Display the report

```{code-cell} ipython3
from IPython.display import HTML

HTML(filename="model-comparison-report.html")
```