```python
# to execute notebooks in parallel
from ploomber import DAG
from ploomber.tasks import NotebookRunner
from ploomber.products import File
from ploomber.executors import Parallel

# to produce parameter grid
from sklearn.model_selection import ParameterGrid

# to create SQLite database
from sklearn_evaluation import NotebookDatabase
```

```python
nb = """
# %% tags=["parameters"]
model = None
params = None

# %%
import importlib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %%
ca_housing = datasets.fetch_california_housing(as_frame=True)
df = ca_housing['frame']

# %%
X = df.drop('MedHouseVal', axis='columns')
y = df.MedHouseVal

# %%
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42)

# %% tags=["model"]
mod, _, attr = model.rpartition('.')
reg = getattr(importlib.import_module(mod), attr)(**params)
reg.fit(X_train, y_train)
print(model)

# %% tags=["params"]
print(reg.get_params())

# %% tags=["mse"]
y_pred = reg.predict(X_test)
mean_squared_error(y_test, y_pred)
"""
```

```python
# create DAG, use the Parallel executor
dag = DAG(executor=Parallel())

experiments = {
    'sklearn.tree.DecisionTreeRegressor': ParameterGrid(dict(criterion=['squared_error', 'friedman_mse'], splitter=['best', 'random'], max_depth=[3, 5])),
    'sklearn.linear_model.Lasso': ParameterGrid(dict(alpha=[1.0, 2.0, 3.0], fit_intercept=[True, False])),
}

# generate one task per set of parameter
for model, grid in experiments.items():
    for i, params in enumerate(grid):
        name = f'{model}-{i}'
        NotebookRunner(nb, File(f'outputs/{name}.ipynb'), dag=dag, ext_in='py', name=name,
                       # the embedded engine is more reliable
                       papermill_params=dict(engine_name='embedded'),
                       params=dict(model=model, params=params))
```

```python
# print summary table
dag.status()
```

```python
# run notebooks in parallel!
dag.build(force=True)
```

```python
# initialize db with notebooks in the outputs directory
db = NotebookDatabase('nb.db', 'outputs/*.ipynb')
db.index()
```

Let's find the notebooks with the lowest error:

```python
# load jupysql magic
%load_ext sql
```

```sql magic_args="sqlite:///nb.db"
SELECT
    path,
    json_extract(c, '$.model') AS model,
    json_extract(c, '$.mse') AS mse
FROM nbs
ORDER BY 3 ASC
LIMIT 3
```

Average error of DecisionTree and Lasso:

```sql
SELECT
    json_extract(c, '$.model') AS model,
    AVG(json_extract(c, '$.mse')) AS avg_mse
FROM nbs
GROUP BY 1
ORDER BY 2 ASC
```

DecisionTree by performance:

```sql
SELECT
    json_extract(c, '$.model') AS model,
    json_extract(c, '$.mse') AS mse,
    json_extract(c, '$.params.max_depth') AS max_depth,
    json_extract(c, '$.params.criterion') AS criterion,
    json_extract(c, '$.params.splitter') AS splitter
FROM nbs
WHERE json_extract(c, '$.model') LIKE '%sklearn.tree.DecisionTreeRegressor%'
ORDER BY mse ASC
```

```python

```
