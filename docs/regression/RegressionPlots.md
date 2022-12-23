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

# Evaluate Regression

+++

These 2 plots allow you to visualize your learning's accuracy.

```{code-cell} ipython3
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn_evaluation import plot
import matplotlib.pyplot as plt
```

## Fetch Data

```{code-cell} ipython3
X, y = load_diabetes(return_X_y=True)
```

```{code-cell} ipython3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

## Conduct Learning

```{code-cell} ipython3
reg = LinearRegression()
reg.fit(X_train, y_train)
```

```{code-cell} ipython3
y_pred = reg.predict(X_test)
y_true = y_test
```

## Visualize Evaluations

+++

### Residuals Plot

+++

This plot shows the residual values' distribution against the predicted value.

```{code-cell} ipython3
plot.residuals(y_true, y_pred)
```

### Prediction Error Plot

+++

This plot has 2 lines: the identity line (where y_predicted=y_measured) and the best fit regression line of y_predicted against y_measured. The difference between the 2 lines visualizes the error of the prediction and trend of learning.

```{code-cell} ipython3
plot.prediction_error(y_true, y_pred)
```
