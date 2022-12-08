#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[1]:


model = None
params = None
upstream = None
product = None


# In[2]:


# Parameters
model = "sklearn.tree.DecisionTreeRegressor"
params = {"criterion": "friedman_mse", "max_depth": 3, "splitter": "best"}
upstream = {"data": {"nb": "output/data.html", "data": "output/data.csv"}}
product = "output/models/sklearn.tree.DecisionTreeRegressor-4.ipynb"


# In[3]:


import importlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[4]:


df = pd.read_csv(upstream['data']['data'])


# In[6]:


X = df.drop('MedHouseVal', axis='columns')
y = df.MedHouseVal


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=0)


# In[8]:


mod, _, attr = model.rpartition('.')
reg = getattr(importlib.import_module(mod), attr)(**params)
reg.fit(X_train, y_train)
print(model)


# In[9]:


print(reg.get_params())


# In[10]:


y_pred = reg.predict(X_test)
mean_squared_error(y_test, y_pred)

