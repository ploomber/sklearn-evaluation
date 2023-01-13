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

# Classification


```{important}
this is some new stuff
```

In this guide we'll show how to compare and evaluate models with sklearn-evaluation.
We will use the penguins dataset and will try to classify based on parameters such as bill and flipper size, and which penguin species is it.

The steps in this guide are:
1. Loading the dataset
2. Data cleaning
3. Fitting models
4. Evaluating the features and models
5. Comparing the different models

In steps 4 & 5 the real value of sklearn-evaluation comes to fruition as we get a lot of visualizations out of the box which will help us compare and evaluate the models, making it easier to pick the optimal one.

```{code-cell} ipython3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn_evaluation import plot, table
# Based on https://github.com/Adeyinka-hub/Machine-Learning-2/blob/master/Penguin%20Dataset.ipynb
```

## Load the dataset

```{code-cell} ipython3
df = sns.load_dataset('penguins')

# Review a sample of the data
df.head(5)
```

+++ {"tags": []}

## Data cleaning

In this section, we're cleaning and preparing the dataset for fitting.
It's all in a single cell since this isn't too relevant to the tool itself.

```{code-cell} ipython3
df.isnull().sum()
df.dropna(inplace=True)
Y = df.species
Y = Y.map({'Adelie': 0, 'Chinstrap':1, 'Gentoo':2})
df.drop('species', inplace=True, axis=1)
se=pd.get_dummies(df['sex'], drop_first=True) 
df = pd.concat([df, se], axis=1)
df.drop('sex', axis=1, inplace=True)
le = LabelEncoder()
df['island']= le.fit_transform(df['island']) 
```

# Decision Tree Classifier

```{code-cell} ipython3
X = df
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=40)


dtc = tree.DecisionTreeClassifier()
dt_model = dtc.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("Acc on test data: {:,.3f}".format(dtc.score(X_test, y_test)))
```

```{code-cell} ipython3
y_test
{'Adelie': 0, 'Chinstrap':1, 'Gentoo':2}
```

## Evaluate our model

In this section, we can easily evaluate our model via a confusion matrix, and understand which feature affects our accuracy by order of importance.

```{code-cell} ipython3
plot.confusion_matrix(y_test, y_pred_dt)
```

```{code-cell} ipython3
plot.feature_importances(dtc, top_n=5, feature_names=list(dtc.feature_names_in_))
```

In addition to the plot, we can also represent the importance through a table, which we can later track and query via SQL. For more information, check our [tracking guide](https://sklearn-evaluation.ploomber.io/en/latest/api/SQLiteTracker.html)

```{code-cell} ipython3
print(table.feature_importances(dtc, feature_names=list(dtc.feature_names_in_)))
```

# KNN classifier

```{code-cell} ipython3
:tags: []

KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
y_pred_knn= KNN.predict(X_test)
print(accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
knn_cm = plot.confusion_matrix(y_test, y_pred_knn)
```

# Comparing KNN and Random Forest reports

In this section, we will overlay both of the models via the confusion matrices. We will do the same with the classification report. This will allow us to pick the superior model without a lot of effort.

```{code-cell} ipython3
knn_cm = plot.ConfusionMatrix.from_raw_data(y_test, y_pred_knn)
dt_cm = plot.ConfusionMatrix.from_raw_data(y_test, y_pred_dt)
```

```{code-cell} ipython3
knn_cm + dt_cm
```

```{code-cell} ipython3
dt_cr = plot.ClassificationReport.from_raw_data(y_test, y_pred_dt)
knn_cr = plot.ClassificationReport.from_raw_data(y_test, y_pred_knn)
```

```{code-cell} ipython3
knn_cr + dt_cr
```

```{code-cell} ipython3

```
