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

# Basic plots

```{code-cell} ipython3
# Import neccesary libraries

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn_evaluation import plot, table
import matplotlib.pyplot as plt
```

sklearn-evluation has two main modules for evaluating classifiers: [sklearn_evaluation.plot](../api/plot.rst) and [sklearn_evaluation.table](../api/table.rst), let’s see an example of how to use them.

First, let’s load some data and split it in training and test set.

```{code-cell} ipython3
data = datasets.make_classification(200, 10, n_informative=5,
                                    class_sep=0.65)
```

```{code-cell} ipython3
X = data[0]
y = data[1]
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

Now, we are going to train the data using one of the scikit-learn classifiers.
In these case a random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

The model parameters are:
n_estimators=The number of trees in the forest.




```{code-cell} ipython3
est = RandomForestClassifier(n_estimators=5)
est.fit(X_train, y_train)
```

Most of the functions require us to pass the class predictions for the test set (``y_pred``), the scores assigned (``y_score``) and the ground truth classes (``y_true``), let’s define such variables.

```{code-cell} ipython3
y_pred = est.predict(X_test)
y_score = est.predict_proba(X_test)
y_true = y_test
```

We can start evaluating our model, the following example shows how to plot a confusion matrix.

```{code-cell} ipython3
plot.confusion_matrix(y_true, y_pred)
```

Some classifiers (such as [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)) have feature importances, we can plot then passing the estimator object to the feature_importances function.

```{code-cell} ipython3
plot.feature_importances(est, top_n=5)
```

A feature importances function is also available in the table module.

Feature Importance refers to techniques that calculate a score for all the input features for a given model the scores simply represent the importance  of each feature:




```{code-cell} ipython3
print(table.feature_importances(est))
```

A higher score means that the specific feature will have a larger effect on the model that is being used to predict a certain variable, in these case the feature 10 is the most relevant , in order to predict y, wich is the independent variable.

Using sklearn evaluate is a good approach to see the feature importance in a classification task.

Now, let’s see how to generate two of the most common plots for evaluating classifiers: Precision-Recall and ROC.

```{code-cell} ipython3
plot.precision_recall(y_true, y_score)
```

Precision is one indicator of a machine learning model's performance,the quality of a positive prediction made by the model. 
Precision refers to the number of true positives divided by the total number of positive predictions between 0 and 1.


```{code-cell} ipython3
plot.roc(y_true, y_score)
```

The ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds, the function receive two elements y_true and y_scor to measure the ability of the model of distinguish the differencess between classes.

These curve plots elements:

True Positive Rate
False Positive Rate

Using  sklearn evaluate you can plot the features without using the common visualization libraries in python such as matplotlib, and gives the user the efficiency of the model in just one line code.

