# Comparing classifiers

*Added in sklearn-evaluation version 0.7.2*

Learn how to easily compare confusion matrices from different models.

- Compare two models by plotting all values: `cm_1 + cm_2`
- Compare the performance between two models: `cm_1 - cm_2`

Note that only `ConfusionMatrix` has been implemented, if you want us to implement other plots, let us know by sharing your feedback in our [community.](https://ploomber.io/community)

```python
import matplotlib
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn_evaluation import plot
```

```python
matplotlib.rcParams["figure.figsize"] = (7, 7)
matplotlib.rcParams["font.size"] = 20
```

```python
# get training and testing data
X, y = datasets.make_classification(
    1000, 20, n_informative=10, class_sep=0.80, n_classes=3, random_state=0
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def fit_and_plot(est):
    """Fit a model and return a confusion matrix
    """
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    return plot.ConfusionMatrix(y_test, y_pred, normalize=False)


# fit decision tree and random forest, return confusion matrices
tree, forest = [
    fit_and_plot(est) for est in [DecisionTreeClassifier(), RandomForestClassifier()]
]
```

## Confusion matrices

### Decision tree

```python
tree
```

### Random forest

```python
forest
```

## Compare decision tree and random forest

```python
tree + forest
```

## How better is the random forest than the tree?

```python
forest - tree
```
