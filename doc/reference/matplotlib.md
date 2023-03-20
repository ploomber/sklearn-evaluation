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
html_meta:
  "description lang=en": "Explore advanced usage of sklearn-evaluation with matplotlib for enhanced visualizations, customization, and evaluation of ML models with style & control."
  "keywords": "advanced usage, sklearn-evaluation, matplotlib, visualizations, customization, evaluation, machine learning models, plot style, ROC curve, grid plots"
  "property=og:locale": "en_US"
---

# Advanced usage using matplotlib

```{code-cell} ipython3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn_evaluation import plot

import matplotlib.pyplot as plt

data = datasets.make_classification(
    n_samples=200, n_features=10, n_informative=5, class_sep=0.7
)
X = data[0]
y = data[1]
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
est = RandomForestClassifier(n_estimators=10)
est.fit(X_train, y_train)
y_true = y_test
y_score = est.predict_proba(X_test)
est = RandomForestClassifier(n_estimators=3)
est.fit(X_train, y_train)
y_score2 = est.predict_proba(X_test)
```

As we mentioned in the previous section, using the functional interface provides great flexibility to evaluate your models, this sections includes some recipes for common tasks that involve the use of the matplotlib API.

+++

## Changing plot style

+++

sklearn-evaluation uses whatever configuration matplotlib has, if you want to change the style of the plots easily you can use one of the many styles available:

```{code-cell} ipython3
import matplotlib.style

matplotlib.style.available
```

Then change the style using

```{code-cell} ipython3
matplotlib.style.use("ggplot")
```

Let's see how a ROC curve looks with the new style:

```{code-cell} ipython3
plot.roc(y_true, y_score)
```

```{code-cell} ipython3
matplotlib.style.use("classic")
```

## Saving plots

```{code-cell} ipython3
ax = plot.roc(y_true, y_score)
fig = ax.get_figure()
fig.savefig("my-roc-curve.png")
```

```{code-cell} ipython3
import os

os.remove("my-roc-curve.png")
```

## Comparing several models with one plot

```{code-cell} ipython3
fig, ax = plt.subplots()
plot.roc(y_true, y_score, ax=ax)
plot.roc(y_true, y_score2, ax=ax)
ax.legend(["Model 1", "Baseline", "Model 2"])
```

## Grid Plots

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
plot.roc(y_true, y_score, ax=ax1)
plot.roc(y_true, y_score2, ax=ax2)
ax1.legend(["Model 1"])
ax2.legend(["Model 2"])
```

## Customizing plots

```{code-cell} ipython3
ax = plot.roc(y_true, y_score)
ax.set_title("This is a custom title")
```
