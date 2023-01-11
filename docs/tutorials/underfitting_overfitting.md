---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# How to deal with underfitting and overfitting?

+++

In this tutorial we'll cover a common challenge in ML, and how to fix it.
We will understand what is underfitting, overfitting and the difference between the two. 
Once we cover the basics, we will dive deep into:
1. Detect we have an under/overfitting (and understand which one do we have).
2. Understand how to solve it, and if our method solved it.

We will see how the different plots on our disposal (via sklearn-evaluation) can shorten our time to a working model.

**TLDR: A learning curve shows the performance of a model on training and validation data as the amount of data used to train the model increases.** 

**This will indicate if the model is suffering from overfitting or underfitting.**

+++

## Overfitting

+++

Overfitting occurs when a machine learning model is trained too well on the training data.
As a result, it performs poorly on new, unseen data. There are a few signs that a model may be overfitting:
1. The model's performance on training data is significantly better than its performance on the validation or test data.
2. The model's performance flattens or starts to decrease after a certain point in the training process.
3. The model's complexity is significantly greater than the complexity of the problem.

**TODO: Add correlating learning curve**

+++

## Underfitting

+++

Underfitting happens when our model cannot capture the underlying patterns in the training data.
This resulting in poor performance on both the training and validation/test sets. 

There are a few signs that a model may be underfitting:
1. The model's performance on the training data is significantly worse than expected or worse than a simple baseline model.
2. The model's performance on the validation/test data is not much better than random guessing.
3. The model's complexity is significantly lower than the complexity of the problem.

**TODO: Add correlating learning curve**

+++

## Detection

In this section we'll detect which one we're having and what does the learning curve tells us?

+++

## How to solve it?

We'll talk about a few ways we can improve our performance back. As some solutions can help with both and some are relevant only to one set of issues, we'll seperate the methods into 2 sections.


### Overfitting
Regularization, dropout, early stopping, cross-validation, data augmentation, ensemble methods.

### Underfitting:
Increasing the model's complexity, Adding/Collecting more data, Using more informative features, ensemble methods, hyperparameter tuning, transfer learning.


+++

# Conclusion

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

### Cooks Distance

+++

Cooks distance is an effective tool to measure the influence of an outlier in the training dataset for a regression problem. Outliers are data points that vary significantly from the rest of the data points in the training set. The presence of outliers in the training phase can affect the parameters that the model learns. This implementation assumes the Ordinary Least Squares regression.

+++

Create a dataset with strong outliers.

```{code-cell} ipython3
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, 
                                 n_features=6, 
                                 n_informative=5, 
                                 n_targets=1, 
                                 bias=100.0,
                                 noise=30.0)
```

```{code-cell} ipython3
plot.cooks_distance(X,y)
```
