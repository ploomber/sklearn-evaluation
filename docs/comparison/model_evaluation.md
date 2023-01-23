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

# Model evaluation - example

+++


```{code-cell} ipython3
from sklearn_evaluation.models import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import urllib.request
import pandas as pd
```

```{code-cell} ipython3
  # download dataset. Reference: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
  urllib.request.urlretrieve(
      'https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv', filename='./output/heart.csv')

  data = pd.read_csv('./output/heart.csv')

  column = 'fbs'
  X = data.drop(column, axis=1)
  y = data[column]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=2023)

  model = RandomForestClassifier()
  model.fit(X_train, y_train)
```

```{code-cell} ipython3
  y_pred = model.predict(X_test)
  y_score = model.predict_proba(X_test)
```

```
  report = evaluate_model(y_test, y_pred, model=model, y_score=y_score)
```

```{raw} html
:file: ./output/model-evaluation-report.html
```