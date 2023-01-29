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

In this tutorial we will demonstrate how to quickly train, test and compare two models in order to find which one performs better.

We'll be using the heart disease dataset and compare `RandomForestClassifier` to `DecisionTreeClassifier`.

You can download the dataset from <a href="https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data" target="_blank">here</a>.

## Download the data

```{code-cell} ipython3

import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split


urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv', filename='heart.csv')

data = pd.read_csv('heart.csv')
```

## Prepare the data 
```{code-cell} ipython3

column = "target"
X = data.drop(column, axis=1)
y = data[column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2023
)

```

## Model A

For our first model we will use the RandomForestClassifier

```{code-cell} ipython3
:tags: [remove-output]

from sklearn.ensemble import RandomForestClassifier

model_a = RandomForestClassifier()
model_a.fit(X_train, y_train)
```

## Model B

The second model will be based on the DecisionTreeClassifier

```{code-cell} ipython3
:tags: [remove-output]

from sklearn.tree import DecisionTreeClassifier

model_b = DecisionTreeClassifier()
model_b.fit(X_train, y_train)
```

## Compare models

Now, let's use the `compare_models` function and generate our report.
Please note that our report will be more detailed as we provide more parameters.

```{code-cell} ipython3
:tags: [remove-output]

from sklearn_evaluation.models import compare_models

report = compare_models(model_a, model_b, X_train, X_test, y_test)
```

## Display the report

You can embed your report or save it locally using `report.save("report.html")`

```{code-cell} ipython3
from IPython.display import HTML
HTML(report._repr_html_())
```
