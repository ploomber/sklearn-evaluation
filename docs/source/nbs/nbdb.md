# Querying notebooks with SQL

*Added in sklearn-evaluation version `0.6`*. Questions? [Join our community!](https://ploomber.io/community)

<!-- #region -->
`NotebookDatabase` indexes outputs from a collection of notebooks in a SQLite database so you can query them. Any tagged cells will be captured and indexed by the database.

Requirements:

```sh
pip install scikit-learn sklearn-evaluation ploomber jupysql
```

<!-- #endregion -->

```python
from pathlib import Path

# to train models in parallel
from ploomber import DAG
from ploomber.tasks import NotebookRunner
from ploomber.products import File
from ploomber.executors import Parallel, Serial

# to produce parameter grid
from sklearn.model_selection import ParameterGrid

# to create SQLite database
from sklearn_evaluation import NotebookDatabase
```

## Code

`NotebookDatabase` indexes the output of tagged cells. In this example, we're using Python scripts (and tag cells using `# %% tags=["some-tag"]`), but the same concept applies for notebooks (`.ipynb`), [see here](https://docs.ploomber.io/en/latest/user-guide/faq_index.html#parameterizing-notebooks) to learn how to tag cells in `.ipynb` files.

```python
# data loading script
data = """
# %% tags=["parameters"]
upstream = None
product = None

# %%
from sklearn import datasets

# %%
ca_housing = datasets.fetch_california_housing(as_frame=True)
df = ca_housing['frame']
df.to_csv(product['data'], index=False)
"""
Path('data.py').write_text(data)

# model fitting script
model = """
# %% tags=["parameters"]
model = None
params = None
upstream = None
product = None

# %%
import importlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %%
df = pd.read_csv(upstream['data']['data'])

# %%
X = df.drop('MedHouseVal', axis='columns')
y = df.MedHouseVal

# %%
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=0)

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
Path('model.py').write_text(model)
```

## Pipeline declaration

Create a pipeline using [Ploomber](https://docs.ploomber.io/en/latest/) and execute it in parallel.

Note that if your models don't take long to run, using the `Serial` executor might be faster, since spinning up a new subprocess is expensive.

Each experiment will create an output `.ipynb` file.

```python
parallel = True

if parallel:
    executor = Parallel()
else:
    executor = Serial(build_in_subprocess=False)


dag = DAG(executor=executor)


experiments = {
    'sklearn.tree.DecisionTreeRegressor': ParameterGrid(dict(criterion=['squared_error', 'friedman_mse'], splitter=['best', 'random'], max_depth=[3, 5])),
    'sklearn.linear_model.Lasso': ParameterGrid(dict(alpha=[1.0, 2.0, 3.0], fit_intercept=[True, False])),
    'sklearn.linear_model.Ridge':ParameterGrid(dict(alpha=[1.0, 2.0, 3.0], fit_intercept=[True, False])), 
    'sklearn.linear_model.ElasticNet': ParameterGrid(dict(alpha=[1.0, 2.0, 3.0], fit_intercept=[True, False])), 
}

papermill_params=dict(engine_name='embedded', progress_bar=False)

# the embedded engine is more reliable
task_data = NotebookRunner(Path('data.py'), {'nb': File('output/data.html'), 'data': File('output/data.csv')},
               dag=dag, papermill_params=papermill_params)

# generate one task per set of parameter
for model, grid in experiments.items():
    for i, params in enumerate(grid):
        name = f'{model}-{i}'
        task = NotebookRunner(Path('model.py'), File(f'output/models/{name}.ipynb'), dag=dag, name=name,
                       papermill_params=papermill_params,
                       params=dict(model=model, params=params))
        task_data >> task
```

## Pipeline execution

```python
# total experiments to run
len(dag)
```

```python
# run experiments
dag.build(force=True)
```

## Indexing notebooks

```python
# initialize db with notebooks in the outputs directory
db = NotebookDatabase('nb.db', 'output/models/*.ipynb')

# Note: pass update=True if you want to update the database if
# the output notebook changes
db.index(verbose=True, update=False)
```

*Note: the `update` argument in `index()` was added in sklearn-evaluation version `0.7`*


## Querying notebooks

`NotebookDatabase` uses SQLite. Here we use [JupySQL](https://jupysql.readthedocs.io/en/latest/intro.html) to query our experiments.

```python
# load jupysql magic
%load_ext sql
```

### Best performing models

```sql magic_args="sqlite:///nb.db"
SELECT
    path,
    json_extract(c, '$.model') AS model,
    json_extract(c, '$.mse') AS mse
FROM nbs
ORDER BY 3 ASC
LIMIT 3
```

<!-- #region -->
*Note:* If using SQLite 3.38.0 (which ships with Python >=3.10) or higher, you can use the shorter `->>` operator:

```sql
SELECT
    path,
    c ->> '$.model' AS model,
    c ->> '$.mse' AS mse
FROM nbs
ORDER BY 3 ASC
LIMIT 3
```

See SQLite's [documentation](https://www.sqlite.org/json1.html#jptr) for details.
<!-- #endregion -->

### Average error by model type

```sql
SELECT
    json_extract(c, '$.model') AS model,
    AVG(json_extract(c, '$.mse')) AS avg_mse
FROM nbs
GROUP BY 1
ORDER BY 2 ASC
```

### DecisionTree by performance

```sql
SELECT
    json_extract(c, '$.model') AS model,
    json_extract(c, '$.mse') AS mse,
    json_extract(c, '$.params.max_depth') AS max_depth,
    json_extract(c, '$.params.criterion') AS criterion,
    json_extract(c, '$.params.splitter') AS splitter
FROM nbs
WHERE json_extract(c, '$.model') = 'sklearn.tree.DecisionTreeRegressor'
ORDER BY mse ASC
LIMIT 5
```
