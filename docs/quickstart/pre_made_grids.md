---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Pre made grids

When training a new model, we often want to tune its hyperparameters; however, defining the grid of values is time-consuming since we need to look at the parameters, read the docs and define the values we want to try. 

In this guide we will show you how to automate this process with our pre made grids and quickly find the optimal set of hyperparameters using our RandomForestClassifierGrid.

## Prepare the data
First things first, let's prepare our data.

Here we are going to use the HeartDiseasesUCI dataset.

```{code-cell} ipython3
import urllib.request
import pandas as pd 

# download dataset. Reference: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
urllib.request.urlretrieve('https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv', filename='heart.csv')

data = pd.read_csv('heart.csv')

data.head()
```

## Specify variables

```{code-cell} ipython3
X = data.drop('target', axis = 1)

y = data['target']
```

## Split the data

```{code-cell} ipython3
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2023)
```

## Initialize the model

Create a new `RandomForestClassifierGrid` and set which grid size you would like to run. 
You can choose any of the following sizes: `tiny`, `small`, `medium`, `large` or `xlarge`.

For this example we will use a medium grid.

```{code-cell} ipython3
from sklearn_evaluation.grid import RandomForestClassifierGrid

model = RandomForestClassifierGrid(grid='medium')
```

## Train all models

```{code-cell} ipython3
model.fit(X_train, y_train)
```


# Evaluate

`RandomForestClassifierGrid` uses `GridSearchCV` under the hood, we can easily access it via the `grid_search_cv_` attribute.

```{code-cell} ipython3
model.grid_search_cv_.best_params_
```

## Set test data

After the training is done we can set the test data, and evaluate our model using our out-of-the-box plots.

Please note that at the moment the plotting methods are using the GridSearchCV `best_estimator_ `.

```{code-cell} ipython3
model.set_test_data(X_test, y_test)
```

## Plots

## Confusion matrix

```{code-cell} ipython3
model.confusion_matrix()
```

## ROC

```{code-cell} ipython3
model.roc()
```

## Feature importances

```{code-cell} ipython3
model.feature_importances()
```

## Grid search

```{code-cell} ipython3
model.grid_search()
```