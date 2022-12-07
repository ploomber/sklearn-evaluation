#!/usr/bin/env python
# coding: utf-8

# 
# [![open-in-jupyterlab](https://raw.githubusercontent.com/ploomber/ploomber/master/_static/open-in-jupyterlab.svg)](https://binder.ploomber.io/v2/gh/ploomber/binder-env/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fploomber%252Fsklearn-evaluation%26urlpath%3Dlab%252Ftree%252Fsklearn-evaluation%252Fdocs/source/nbs/compare.ipynb%26branch%3Dmaster)
# 
# Or try locally: `pip install k2s -U && k2s get ploomber/sklearn-evaluation/master/docs/source/nbs/compare.ipynb`
# 

# # Comparing classifiers
# 
# Learn how to easily compare plots from different models.
# 
# - Compare two models by plotting all values: `plot1 + plot2`
# - Compare the performance between two models: `plot2 - plot1`
# 
# ## Confusion matrix
# 
# *Added in sklearn-evaluation version 0.7.2*

# In[1]:


import matplotlib
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn_evaluation import plot


# In[2]:


matplotlib.rcParams["figure.figsize"] = (7, 7)
matplotlib.rcParams["font.size"] = 18


# In[3]:


# get training and testing data
X, y = datasets.make_classification(
    1000, 20, n_informative=10, class_sep=0.80, n_classes=3, random_state=0
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# fit decision tree and random forest, return confusion matrices
tree_pred, forest_pred = [
    est.fit(X_train, y_train).predict(X_test)
    for est in [DecisionTreeClassifier(), RandomForestClassifier()]
]

tree_cm = plot.ConfusionMatrix(y_test, tree_pred, normalize=False)
forest_cm = plot.ConfusionMatrix(y_test, forest_pred, normalize=False)


# ### Decision tree confusion matrix

# In[4]:


tree_cm


# ### Random forest confusion matrix

# In[5]:


forest_cm


# ### Compare confusion matrices

# In[6]:


tree_cm + forest_cm


# In[7]:


forest_cm - tree_cm


# ## Classification report
# 
# *Added in sklearn-evaluation version 0.7.8*

# In[8]:


tree_cr = plot.ClassificationReport(y_test, tree_pred)
forest_cr = plot.ClassificationReport(y_test, forest_pred)


# ### Decision tree classification report

# In[9]:


tree_cr


# ### Random forest classification report

# In[10]:


forest_cr


# ### Compare classification reports

# In[11]:


tree_cr + forest_cr


# In[12]:


forest_cr - tree_cr

