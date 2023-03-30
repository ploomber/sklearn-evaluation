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
    description lang=en: "Learn about feature ranking, a technique to evaluate single features or pairs of features in machine learning, with examples in Python using scikit-learn and matplotlib."
    keywords: "feature ranking, machine learning, Python, scikit-learn, matplotlib, 1D rank, 2D rank, Shapiro-Wilk algorithm"
    property=og:locale: "en_US"
notebook_metadata_filter: myst
---

# Feature Ranking

Feature ranking allows to evaluate single features or pairs of features using a variety of metrics that score the features on the scale [-1, 1] or [0, 1] allowing them to be ranked. Two types of ranking are supported right now:

- 1-D Rank : Ranking that considers one feature at a time and plots the relative ranks of each feature on a bar chart. Default is Shapiro-Wilk algorithm.  
- 2-D Rank : Ranking that considers pairs of features at a time and visualizes the ranks on the lower left triangle of a feature co-occurence matrix.

```{code-cell} ipython3
import matplotlib
from sklearn.datasets import load_iris as load_data
from sklearn_evaluation.plot import Rank1D, Rank2D
```

```{code-cell} ipython3
matplotlib.rcParams["figure.figsize"] = (7, 7)
matplotlib.rcParams["font.size"] = 18
```

```{code-cell} ipython3
X, y = load_data(return_X_y=True)
features = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
```

##### Rank 1D

```{code-cell} ipython3
rank1d = Rank1D(features=features)
rank1d.feature_ranks(X)
```

##### Rank 2D

```{code-cell} ipython3
rank2d = Rank2D(features=features)
rank2d.feature_ranks(X)
```
