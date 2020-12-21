"""
Tools for comparing model results
"""

# Things to compare:
#
# * Single metrics, all of them can be expressed as a single dictionary, then each experiment is a row and each metric a column
#     * Perform a diff operation (e.g. difference or ratio)
#     * As styled table
#     * Visually
#     * if its just two, add diff column, otherwise display with colors the max and min
# * Parameters, also represented by dicts
#     * But the difference is presented as a list
#     * Like, A, B C parameters are thge same, C is different, here are the values
# * Sliced metrics
#     * Like the first one but each metric is broken down by some other factor
#     * Difference could be represented by the difference of the table but we could only compare two at a time
# * Plots
#     * Add a tab to switch between them

# %load_ext autoreload
# %autoreload 2

# +
import papermill as pm
import jupytext

from sklearn_evaluation import NotebookIntrospector, NotebookCollection
# -

nb = jupytext.read('train.py')
jupytext.write(nb, 'train.ipynb')

# +
models = ['sklearn.ensemble.RandomForestRegressor', 'sklearn.linear_model.LinearRegression', 'sklearn.svm.LinearSVR']

for model in models:
    pm.execute_notebook('train.ipynb', f'{model}.ipynb', parameters={'model': model})

# +
files = [f'{model}.ipynb' for model in models]
ids = ['rf', 'lr', 'svr']

col = NotebookCollection(files[:3], keys=ids)
# -

col['model_name']

col['plot']

col['metrics']

col['metrics_dict']

col['river']


