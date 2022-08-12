```python
from pathlib import Path

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
Path('model.py').write_text(model)
```

```python
# create DAG, use the Parallel executor
dag = DAG(executor=Parallel())
# dag = DAG()

experiments = {
    'sklearn.tree.DecisionTreeRegressor': ParameterGrid(dict(criterion=['squared_error', 'friedman_mse'], splitter=['best', 'random'], max_depth=[3, 5])),
    'sklearn.linear_model.Lasso': ParameterGrid(dict(alpha=[1.0, 2.0, 3.0], fit_intercept=[True, False])),
    'sklearn.linear_model.Ridge':ParameterGrid(dict(alpha=[1.0, 2.0, 3.0], fit_intercept=[True, False])), 
    'sklearn.linear_model.ElasticNet': ParameterGrid(dict(alpha=[1.0, 2.0, 3.0], fit_intercept=[True, False])), 
}

task_data = NotebookRunner(Path('data.py'), {'nb': File('output/data.html'), 'data': File('output/data.csv')},
               dag=dag)

# generate one task per set of parameter
for model, grid in experiments.items():
    for i, params in enumerate(grid):
        name = f'{model}-{i}'
        task = NotebookRunner(Path('model.py'), File(f'output/models/{name}.ipynb'), dag=dag, name=name,
                       # the embedded engine is more reliable
                       papermill_params=dict(engine_name='embedded'),
                       params=dict(model=model, params=params))
        task_data >> task
```

```python
# total experiments to run
len(dag)
```

```python
# run experiments in parallel!
dag.build()
```

```python
# initialize db with notebooks in the outputs directory
db = NotebookDatabase('nb.db', 'output/models/*.ipynb')
db.index(verbose=False)
```

Let's find the notebooks with the lowest error:

```python
# load jupysql magic
%load_ext sql
```

Find best performing models:

```sql magic_args="sqlite:///nb.db"
SELECT
    path,
    json_extract(c, '$.model') AS model,
    json_extract(c, '$.mse') AS mse
FROM nbs
ORDER BY 3 ASC
LIMIT 3
```

Average error by model type:

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
WHERE json_extract(c, '$.model') = 'sklearn.tree.DecisionTreeRegressor'
ORDER BY mse ASC
```

```python

```
