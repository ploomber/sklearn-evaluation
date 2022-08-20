```python
%load_ext autoreload
%autoreload 2
```

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
# generate synthetic data
data = datasets.make_classification(
    1000, 20, n_informative=10, class_sep=0.80, n_classes=3, random_state=0
)
X, y = data

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def fit_and_plot(est):
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

```python

```
