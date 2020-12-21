# %load_ext autoreload
# %autoreload 2

# # Analyzing results from notebooks
#
# The `.ipynb` format is capable of holding tables and charts in a standalone file. This makes it a great choice for model evaluation reports. `NotebookCollection` allows you to retrieve results from previously executed notebooks to quickly compare the differences.

# +
import papermill as pm
import jupytext

from sklearn_evaluation import NotebookIntrospector, NotebookCollection
# -

# Let's first generate a few notebooks, we have a `train.py` script that contains code for training a model, let's convert that to a jupyter notebook using jupytext:

nb = jupytext.read('train.py')
jupytext.write(nb, 'train.ipynb')

# Now, we use papermill to execute the notebook with different parameters, we'll run 4 models: 2 random forest, a linear regression and a support vector regression:

# +
# models with their corresponding parameters
params = [
    {'model': 'sklearn.ensemble.RandomForestRegressor', 'params': {'n_estimators': 50}},
    {'model': 'sklearn.ensemble.RandomForestRegressor', 'params': {'n_estimators': 100}},
    {'model': 'sklearn.linear_model.LinearRegression', 'params': {'normalize': True}},
    {'model': 'sklearn.svm.LinearSVR', 'params': {}},
]

# ids to identify each experiment
keys = ['random_forest_1', 'random_forest_2', 'linear_regression', 'support_vector_regression']

# output files
files = [f'{k}.ipynb' for k in keys]

# execute notebooks using papermill
for f, p in zip(files, params):
    pm.execute_notebook('train.ipynb', output_path=f, parameters=p, progress_bar=False)
# -

# Let's now use `NotebookCollection` to compare results, we just have to pass a list of files to use, and optionally, ids for each notebook (uses the path by default). The only requirement for this to work is to tag cells in the notebooks, each tag then becomes a key in the notebook collection. (To activate the tags menu click on `View -> Cell toolbar -> Tags`).

nbs = NotebookCollection(paths=files, keys=keys)

# To get a list of tags available:

list(nbs)

# We can get dictionaries with model parameters, click on the tabs to switch:

# Pro-tip: then typing the tag, press the "Tab" key for autocompletion!
nbs['model_params']

# And plots:

nbs['plot']

# Tables are also available. A "Summary" tab is added showing all results at once with colors to identify the minimum (green) error value and maximum (red):

nbs['metrics']

# If instead of single-row tables, each experiment generates tables with multiple rows, the summary is not generated:

nbs['river']

# When comparing two, notebooks there are a few differences:

# only compare two notebooks
nbs = NotebookCollection(paths=files[:2], keys=keys)

# The summary that compares single-row tables includes a diff columns:

nbs['metrics']

# When comparing multi-row tables, the Summary tab appears, showing the difference between the tables:

nbs['river']

# Dictionaries generate a Summary tab with the differences:

nbs['model_params']

# ## Using the mapping interface
#
# `NotebookCollection` objects have a dict-like interface, you can retrieve data from individual notebooks:

nbs['model_params']['random_forest_1']

nbs['plot']['random_forest_2']
