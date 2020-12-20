# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import importlib
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import pandas as pd

# + tags=["parameters"]
model = 'sklearn.ensemble.RandomForestRegressor'
# -

d = load_boston()
X = d['data']
y = d['target']

tokens = model.split('.')
name, module = tokens[-1], '.'.join(tokens[:-1])
class_ = getattr(importlib.import_module(module), name)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = class_()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# + tags=["plot"]
sns.scatterplot(x=y_pred, y=y_test)
# -

metrics_ = {
    'mae': metrics.mean_absolute_error(y_test, y_pred),
    'mse': metrics.mean_squared_error(y_test, y_pred)
}

# + tags=["metrics"]
pd.DataFrame(metrics_, index=[0])
