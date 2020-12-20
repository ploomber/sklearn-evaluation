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

import papermill as pm
import jupytext

nb = jupytext.read('train.py')
jupytext.write(nb, 'train.ipynb')

# +
models = ['sklearn.ensemble.RandomForestRegressor', 'sklearn.linear_model.LinearRegression', 'sklearn.svm.LinearSVR']

for model in models:
    pm.execute_notebook('train.ipynb', f'{model}.ipynb', parameters={'model': model})
# -

from sklearn_evaluation import NotebookIntrospector, NotebookCollection

# +
files = [f'{model}.ipynb' for model in models]
ids = ['rf', 'lr', 'svr']

col = NotebookCollection(files, keys=ids, to_df=True)
# -

col['plot']

col['metrics']

col['metrics_dict']

# +
# maybe add a less is better option to color code green/red
m1 = {'mse': 4.0, 'mae': 2.0, 'max_error': 5.5}
m2 = {'mse': 3.5, 'mae': 1.9, 'max_error': 10.1}
m3 = {'mse': 2.5, 'mae': 0.9, 'max_error': 5.1}

# ids = ['exp1', 'exp2', 'exp3']

# +
# out = metrics_df([m1, m2, m3], ids)
# out
# -

d1 = {'mse': {'group1': 4.5, 'group2': 5.5}, 'mae': {'group1': 2.2, 'group2': 1.5}}
d2 = {'mse': {'group1': 5.8, 'group2': 3.2}, 'mae': {'group1': 3.5, 'group2': 2.1}}

df1 = pd.DataFrame(d1).T
df2 = pd.DataFrame(d2).T








