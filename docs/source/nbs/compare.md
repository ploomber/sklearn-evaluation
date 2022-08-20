```python
%load_ext autoreload
%autoreload 2
```

```python
import matplotlib.pyplot as plt
import matplotlib
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn_evaluation import plot

```

```python
matplotlib.rcParams['figure.figsize'] = (6, 6)
matplotlib.rcParams['font.size'] = 18
```

```python
# datasets.load_iris()
```

```python
data = datasets.make_classification(1000, 20, n_informative=10, class_sep=0.80, n_classes=2)
X = data[0]
y = data[1]

# data = datasets.load_iris()
# X = data['data']
# y = data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def fit_and_plot(est):
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    return plot.ConfusionMatrix(y_test, y_pred, normalize=False)

estimators = [DecisionTreeClassifier(), SVC()]
tree, svc = [fit_and_plot(est) for est in estimators]
```

```python
tree
```

```python
svc
```

```python
tree + svc
```

```python
tree - svc
```

```python

```
