#!/usr/bin/env python
# coding: utf-8

# # Evaluate Regression

# These 2 plots allow you to visualize your learning's accuracy.

# In[1]:


from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn_evaluation import plot
import matplotlib.pyplot as plt


# ## Fetch Data

# In[2]:


X, y = load_diabetes(return_X_y=True)


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# ## Conduct Learning

# In[4]:


reg = LinearRegression()
reg.fit(X_train, y_train)


# In[5]:


y_pred = reg.predict(X_test)
y_true = y_test


# ## Visualize Evaluations

# ### Residuals Plot

# This plot shows the residual values' distribution against the predicted value.

# In[6]:


plot.residuals(y_true, y_pred)


# ### Prediction Error Plot

# This plot has 2 lines: the identity line (where y_predicted=y_measured) and the best fit regression line of y_predicted against y_measured. The difference between the 2 lines visualizes the error of the prediction and trend of learning.

# In[7]:


plot.prediction_error(y_true, y_pred)

