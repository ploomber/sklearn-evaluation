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
from sklearn.preprocessing import power_transform
import numpy as np
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

Cooks distance is an effective tool to measure the influence of an outlier in the training dataset for a regression problem. Outliers are data points that vary significantly from the rest of the data points in the training set. The presence of outliers in the training phase can affect the parameters that the model learns. This implementation assumes the Ordinary Least Squares (OLS) regression.

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

## Heteroscedasticity in Regression

+++

When discussing the residuals in regression analysis, we refer to heteroscedasticity, which is characterized by a systematic change in the spread of the residuals over the range of the observed values. The issue with heteroscedasticity is that it violates the assumption of Ordinary Least Squares (OLS) regression that all residuals originate from a population with a uniform variance (homoscedasticity).

### Detecting Heteroscedasticity

+++

First, create a Residuals Plot, using the same code provided [above](#residuals-plot).  

```{code-cell} ipython3
plot.residuals(y_true, y_pred)
```

Ideally, this plot should display random residuals (no patterns) that are uncorrelated and uniform. If you see patterns in the residuals, you might not be able to trust the results of your model. Heteroscedasticity produces a distinctive fan or cone shape in residual plots. In summary, the telltale pattern for heteroscedasticity is that as the predicted values on the x-axis increase, the variance of the residuals also increases. 

### Variable Transformations to Fix Heteroscedasticity

+++

One way to fix heteroscedasticity in the data is to transform either the independent variables (X) or the dependent variable (y). The following code shows how to use a Box-Cox transformation on the dependent variable, y, to fix heteroscedasticity. Note that the **Box-Cox transformation requires the dependent variable to be positive**. If the dependent variable has negative samples, you can use a Yeo-Johnson transformation instead.
  
```{code-cell} ipython3 
# check for negative values in the dependent variable, y
y.min()
```

Because the minimum value of y is positive, we will use the Box-Cox transformation, which **requires y to be a 2D array as input**. After reshaping, using `numpy`, each row represents a sample and each column represents a feature. In this case, y is being treated as a single feature with multiple samples, so it needs to be reshaped accordingly.

```{code-cell} ipython3 
# transform y using the Box-Cox transformation
y_transformed = power_transform(y.reshape(-1, 1), method="box-cox")
```

```{code-cell} ipython3 
# split the dataset again using the transformed y
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.3)
```

```{code-cell} ipython3 
# fit the model
reg.fit(X_train, y_train)
```

```{code-cell} ipython3 
# get predictions
y_pred = reg.predict(X_test)
y_true = y_test
```

```{code-cell} ipython3 
# plot the residuals
plot.residuals(y_true, y_pred)
```
