# # Analyzing results from notebooks
#
# The `.ipynb` format is capable of holding tables and charts in a standalone file. This makes it a great choice for model evaluation reports. `NotebookCollection` allows you to retrieve results from previously executed notebooks to quickly compare the differences.

# +
import papermill as pm
import jupytext

from sklearn_evaluation import NotebookCollection
# -

# Let's first generate a few notebooks, we have a `train.py` script that contains code for training a model, let's convert that to a jupyter notebook using jupytext:

nb = jupytext.read('train.py')
jupytext.write(nb, 'train.ipynb')

# Now, we use papermill to execute the notebook with different parameters, we'll run 4 models: 2 random forest, a linear regression and a support vector regression:

# +
# models with their corresponding parameters
params = [{
    'model': 'sklearn.ensemble.RandomForestRegressor',
    'params': {
        'n_estimators': 50
    }
}, {
    'model': 'sklearn.ensemble.RandomForestRegressor',
    'params': {
        'n_estimators': 100
    }
}, {
    'model': 'sklearn.linear_model.LinearRegression',
    'params': {
        'normalize': True
    }
}, {
    'model': 'sklearn.svm.LinearSVR',
    'params': {}
}]

# ids to identify each experiment
ids = [
    'random_forest_1', 'random_forest_2', 'linear_regression',
    'support_vector_regression'
]

# output files
files = [f'{i}.ipynb' for i in ids]

# execute notebooks using papermill
for f, p in zip(files, params):
    pm.execute_notebook('train.ipynb',
                        output_path=f,
                        parameters=p,
                        progress_bar=False)
# -

# Let's now use `NotebookCollection` to compare results.
#
# We just have to pass a list of files to use, and optionally, ids for each notebook (uses the path by default).
#
# The only requirement for this to work is to tag cells in the notebooks, each tag then becomes a key in the notebook collection. (To activate the tags menu click on `View -> Cell toolbar -> Tags`).
#
# Extracted tables color cells to make it easier to spot the best and worst experiment. By default it assumes that metrics are errors (smaller is better), if you only have score metrics (larger is better), pass `scores=True`, if you have errors and scores, pass a list of scores:

nbs = NotebookCollection(paths=files, ids=ids, scores=['r2'])

# To get a list of tags available:

list(nbs)

# `model_params` contains a dictionary with model parameters, let's get them (click on the tabs to switch):

# Pro-tip: then typing the tag, press the "Tab" key for autocompletion!
nbs['model_params']

# `plot` has a `y_true` vs `y_pred` chart:

nbs['plot']

# If any cell outputs a pandas DataFrame, we can also retrieve it. `metrics` outputs a data frame with a single row where columns are mean absolute error (mae) and mean squared error (mse).
#
# For single-row tables, a "Compare" tab is added showing all results at once with colors to identify the minimum (green) and maximum (red) error values and the other way around for scores:

nbs['metrics']

# We can quickly see that the first random forest is performing the best in both metrics.
#
# Multi-row tables can also be retrieved, `river` contains a table where we have both error metrics broken down by the `CHAS` indicator features. Multi-row tables *do not* display the "Compare" tab:

nbs['river']

# If we only compare two notebooks, the output is a bit different:

# only compare two notebooks
nbs_two = NotebookCollection(paths=files[:2], ids=ids[:2], scores=['r2'])

# The output that compares single-row tables includes a diff column, with the error difference between experiments. Error reductions are showed in green, increments in red:

nbs_two['metrics']

# When comparing multi-row tables, the "Compare" tab appears, showing the difference between the tables:

nbs_two['river']

# When displaying dictionaries, a "Compare" tab shows with a diff view:

nbs_two['model_params']

# Lists (and sets) are compared depending on their elements:

nbs_two['feature_names']

# ## Using the mapping interface
#
# `NotebookCollection` has a dict-like interface, you can retrieve data from individual notebooks:

nbs['model_params']['random_forest_1']

nbs['plot']['random_forest_2']


