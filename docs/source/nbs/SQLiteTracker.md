# Tracking Machine Learning experiments

`SQLiteTracker` provides a simple yet powerful way to track ML experiments using a SQLite database.

```python
from sklearn_evaluation import SQLiteTracker

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```

```python tags=["parameters"]
tracker = SQLiteTracker('my_experiments.db')
```

```python
iris = load_iris(as_frame=True)
X, y = iris['data'], iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

models = [RandomForestRegressor(), LinearRegression(), Lasso()]
```

```python
for m in models:
    model = type(m).__name__
    print(f'Fitting {model}')

    # .new() returns a uuid and creates an entry in the db
    uuid = tracker.new()
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # add data with the .update(uuid, {'param': 'value'}) method
    tracker.update(uuid, {'mse': mse, 'model': model, **m.get_params()})
```

Or use `.insert(uuid, params)` to supply your own ID:

```python
svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

tracker.insert('my_uuid', {'mse': mse, 'model': type(svr).__name__, **svr.get_params()})
```

`tracker` shows last experiments by default:

```python
tracker
```

## Querying experiments

```python
ordered = tracker.query("""
SELECT uuid,
       json_extract(parameters, '$.model') AS model,
       json_extract(parameters, '$.mse') AS mse
FROM experiments
ORDER BY json_extract(parameters, '$.mse') ASC
""")
ordered
```

The query method returns a data frame with "uuid" as the index:

```python
type(ordered)
```

## Adding comments


```python
tracker.comment(ordered.index[0], 'Best performing experiment')
```

User `tracker[uuid]` to get a single experiment:

```python
tracker[ordered.index[0]]
```

## Getting recent experiments

The recent method also returns a data frame:

```python
df = tracker.recent()
df
```

Pass `normalize=True` to convert the nested JSON dictionary into columns:

```python
df = tracker.recent(normalize=True)
df
```

```python
# delete our example database
from pathlib import Path
Path('my_experiments.db').unlink()
```
