#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("/Users/neelashasen/Downloads/credit/credit.csv.gz", compression="gzip")


# In[3]:


data.head()


# In[4]:


import json
with open("/Users/neelashasen/Downloads/credit/meta.json", 'r') as f:
    meta = json.load(f)


# In[5]:


X = data[meta["features"]]
target = data[meta["target"]]


# In[6]:


X.head()


# In[7]:


from scipy.stats import shapiro
ranking_methods = {"shapiro": lambda X: np.array([shapiro(x)[0] for x in X.T])}

def rank(X, algorithm=None):
        """
        Returns the feature ranking.
        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features
        algorithm : str or None
            The ranking mechanism to use, or None for the default
        Returns
        -------
        ranks : ndarray
            An n-dimensional, symmetric array of rank scores, where n is the
            number of features. E.g. for 1D ranking, it is (n,), for a
            2D ranking it is (n,n) and so forth.
        """

        algorithm = algorithm.lower()

        if algorithm not in ranking_methods:
            raise ValueError(
                "'{}' is unrecognized ranking method".format(algorithm)
            )

        # Extract matrix from dataframe if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values

        return ranking_methods[algorithm](X)


# In[8]:


def draw(orientation, ranks, features, show_features_names, ax,**kwargs):
    """
    Draws the bar plot of the ranking array of features.
    """
    if orientation == "h":
        # Make the plot
        ax.barh(np.arange(len(ranks)), ranks, color='g')

        # Add ticks and tick labels
        ax.set_yticks(np.arange(len(ranks)))
        if show_feature_names:
            ax.set_yticklabels(features)
        else:
            ax.set_yticklabels([])

        # Order the features from top to bottom on the y axis
        ax.invert_yaxis()

        # Turn off y grid lines
        ax.yaxis.grid(False)

    elif orientation == "v":
        # Make the plot
        ax.bar(np.arange(len(ranks)), ranks, color=color)

        # Add ticks and tick labels
        ax.set_xticks(np.arange(len(ranks)))
        if show_feature_names:
            ax.set_xticklabels(features, rotation=90)
        else:
            ax.set_xticklabels([])

        # Turn off x grid lines
        ax.xaxis.grid(False)

    else:
        raise ValueError("Orientation must be 'h' or 'v'")
    return ax


# In[9]:


algorithm="shapiro"
orient="h"
show_feature_names=True
features=None
ax=None


# In[10]:


#Fit : class MultiFeatureVisualizer
"""
Parameters
----------
X : array-like, shape (n_samples, n_features)
    Feature dataset to be transformed.
y : array-like, shape (n_samples,)
    Optional dependent target data associated with X.
"""

n_columns = X.shape[1]

if features is not None:
# Use the user-specified features with some checking
# TODO: allow the user specified features to filter the dataset
    if len(features) != n_columns:
        raise ValueError(
            (
                "number of supplied feature names does not match the number "
                "of columns in the training data."
            )
        )

        features_ = np.array(features)

else:
    # Attempt to determine the feature names from the input data
    if isinstance(X, pd.DataFrame):
        features_ = np.array(X.columns)

    # Otherwise create numeric labels for each column.
    else:
        features_ = np.arange(0, n_columns)


# In[ ]:





# In[11]:


ranks = rank(X, algorithm)


# In[12]:


ranks.shape


# In[13]:


import matplotlib.pyplot as plt


if ax is None:
    ax = plt.gca()

ax = draw('h', ranks, features_, True, ax)
ax.set_axisbelow(True)
ax.xaxis.grid(True, color='#808080')
ax.set_title(
            "{} Ranking of {} Features".format(
                algorithm.title(), len(features_)
            )
        )


# In[ ]:





# In[14]:


from sklearn.datasets import load_iris


# In[32]:





# In[15]:


type(iris)


# In[35]:


iris['feature_names']


# In[36]:


data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])


# In[37]:


data.head()


# In[40]:


data2 = pd.DataFrame(data= np.c_[iris['data']], columns= iris['feature_names'])


# In[41]:


data2.head()


# In[38]:


X = data[iris['feature_names']]


# In[39]:


X.head()


# In[ ]:




