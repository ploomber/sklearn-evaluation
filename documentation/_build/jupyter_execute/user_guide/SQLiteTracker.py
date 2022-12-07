#!/usr/bin/env python
# coding: utf-8

# 
# [![open-in-jupyterlab](https://raw.githubusercontent.com/ploomber/ploomber/master/_static/open-in-jupyterlab.svg)](https://binder.ploomber.io/v2/gh/ploomber/binder-env/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fploomber%252Fsklearn-evaluation%26urlpath%3Dlab%252Ftree%252Fsklearn-evaluation%252Fdocs/source/nbs/SQLiteTracker.ipynb%26branch%3Dmaster)
# 
# Or try locally: `pip install k2s -U && k2s get ploomber/sklearn-evaluation/master/docs/source/nbs/SQLiteTracker.ipynb`
# 

# # Experiment tracking
# 
# **Important:** `SQliteTracker` got a big upgrade in version `0.8.2` so ensure you are running such version or a higher one.
# 
# `SQLiteTracker` provides a powerful and flexible way to track computational (e.g., Machine Learning) experiments using a SQLite database. Allows you to use SQL as the query language, giving you a powerful tool for experiment comparison, and comes with plotting features to compare plots side-by-side and to combine plots for better comparison.
# 
# Read more about the motivations in our [blog post](https://ploomber.io/blog/experiment-tracking/), check out the [HN discussion.](https://news.ycombinator.com/item?id=33624018)
# 
# This tutorial will walk you through the features with a Machine Learning use case; however, the tracker is generic enough to be used in any other domains.

# In[1]:


from pathlib import Path

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, RocCurveDisplay


# In[2]:


# delete our example database, if any
db = Path('my_experiments.db')

if db.exists():
    db.unlink()


# In[3]:


X, y = datasets.make_classification(200, 10, n_informative=5, class_sep=0.65)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

models = [RandomForestClassifier(), LogisticRegression(), DecisionTreeClassifier()]


# ## Training and logging models

# In[4]:


for m in models:
    model = type(m).__name__
    print(f'Fitting {model}')


    experiment = tracker.new_experiment()
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # log a dictionary with log_dict
    experiment.log_dict({'accuracy': acc, 'model': model, **m.get_params()})


# Or use `.log(key, value)` to log individual values:

# In[7]:


svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc = accuracy_score(y_test, y_pred)

experiment = tracker.new_experiment()

# log individual values
experiment.log('accuracy', acc)
experiment.log('model', type(svc).__name__)

_ = experiment.log_dict(svc.get_params())


# ## Displaying latest experiments
# 
# Display the `tracker` object to show last experiments:

# In[8]:


tracker


# ## Querying experiments with SQL using `.query()`
# 
# You can use SQL to query your experiments. To see what's been logged, use `get_parameters_keys()`:

# In[9]:


keys = tracker.get_parameters_keys()
# show first 5 keys
keys[:5]


# To generate a sample query, use `.get_sample_query()`:

# In[10]:


print(tracker.get_sample_query())


# To execute a query, use `.query()`:

# In[11]:


ordered = tracker.query("""
SELECT uuid,
       json_extract(parameters, '$.model') AS model,
       json_extract(parameters, '$.accuracy') AS accuracy
FROM experiments
ORDER BY accuracy DESC
""")
ordered


# The query method returns a data frame with "uuid" as the index:

# In[12]:


type(ordered)


# ## Storing plots
# 
# You can log a confusion matrix and classification reports:

# In[13]:


get_ipython().run_cell_magic('capture', '', "\ndef fit(model):\n    model.fit(X_train, y_train)\n    y_pred = model.predict(X_test)\n    acc = accuracy_score(y_test, y_pred)\n    experiment = tracker.new_experiment()\n    experiment.log_dict({'accuracy': acc, 'model': type(model).__name__, **model.get_params()})\n    \n    # log plots     \n    experiment.log_confusion_matrix(y_test, y_pred)\n    experiment.log_classification_report(y_test, y_pred)\n    \n    # log generic matplotlib figure\n    roc = RocCurveDisplay.from_estimator(model, X_test, y_test)\n    experiment.log_figure('roc', roc.figure_)\n    \nfit(model=RandomForestClassifier(n_estimators=100))\nfit(model=RandomForestClassifier(n_estimators=10))\n")


# In[14]:


tracker.recent(2)


# ## Rendering plots in table view
# 
# The `.query()` method also allows rendering plots in the table view:

# In[15]:


results = tracker.query("""
SELECT uuid,
       json_extract(parameters, '$.model') AS model,
       json_extract(parameters, '$.accuracy') AS accuracy,
       json_extract(parameters, '$.confusion_matrix') AS cm,
       json_extract(parameters, '$.roc') AS roc
FROM experiments
ORDER BY created DESC
LIMIT 2
""", as_frame=False, render_plots=True)

results


# ## Side-by-side comparison
# 
# From the `.query()` results, you can extract a given column for a side by side comparison:

# In[16]:


results.get("cm")


# You can change the labels in the tabs with the `index_by` argument:

# In[17]:


results.get("cm", index_by="accuracy")


# ## Combining plots
# 
# With a side-by-side comparison, it might be hard to spot the model performance differents, you can get individual experiments, extract their plots and combine them:

# In[18]:


# get the uuids for the latest 2 experiments
uuid1, uuid2 = tracker.recent(2).index

# get the experiments
one = tracker.get(uuid1)
another = tracker.get(uuid2)


# Combine statistics from both confusion matrices:

# In[19]:


one["confusion_matrix"] + another["confusion_matrix"]


# Get confusion matrix differences:

# In[20]:


one["confusion_matrix"] - another["confusion_matrix"]


# Combine classification reports:

# In[21]:


one["classification_report"] + another["classification_report"]


# You can also retrieve generic figures (logged with `log_figure`):

# In[22]:


one["roc"]


# However, note that plot combination (`plot1 + plot2` and `plot1 - plot2`) is only supported by plots logged via the `experiment.log_*` and not by the generic `experiment.log_figure` method.

# ## Adding comments

# In[23]:


one.comment('This is some comment')


# In[24]:


tracker.query("""
SELECT uuid,
       comment,
       json_extract(parameters, '$.model') AS model,
       json_extract(parameters, '$.accuracy') AS accuracy
FROM experiments
WHERE comment is not NULL
""")


# ## Pandas integration
# 
# ### Getting recent experiments
# 
# The recent method also returns a data frame:

# In[25]:


df = tracker.recent()
df


# Pass `normalize=True` to convert the nested JSON dictionary into columns:

# In[26]:


df = tracker.recent(normalize=True)
df


# ### Querying
# 
# You can also use the `.query()` method with `as_frame=True` (default value) to get a `pandas.DataFrame`

# In[27]:


df = tracker.query("""
SELECT uuid,
       json_extract(parameters, '$.model') AS model,
       json_extract(parameters, '$.accuracy') AS accuracy
FROM experiments
ORDER BY accuracy DESC
LIMIT 3
""")
df


# In[ ]:




