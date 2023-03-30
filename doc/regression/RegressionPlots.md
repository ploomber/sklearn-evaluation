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
    description lang=en: "Evaluate regression models with sklearn-evaluation; visualize residuals, prediction errors, and Cook's distance for outlier influence, using LinearRegression on the diabetes dataset."
    keywords: "evaluate regression, sklearn-evaluation, residual plot, prediction error plot, Cook's distance, visualizations, linear regression, diabetes dataset, outlier influence"
    property=og:locale: "en_US"
notebook_metadata_filter: myst
---

# Evaluate Regression

+++

These 2 plots allow you to visualize your learning's accuracy.

```{code-cell} ipython3
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn_evaluation import plot
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

### Cooks Distance

+++

Cooks distance is an effective tool to measure the influence of an outlier in the training dataset for a regression problem. Outliers are data points that vary significantly from the rest of the data points in the training set. The presence of outliers in the training phase can affect the parameters that the model learns. This implementation assumes the Ordinary Least Squares regression.

+++

Create a dataset with strong outliers.

```{code-cell} ipython3
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=100, n_features=6, n_informative=5, n_targets=1, bias=100.0, noise=30.0
)
```

```{code-cell} ipython3
plot.cooks_distance(X, y)
```
