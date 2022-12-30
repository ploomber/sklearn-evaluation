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

# Regression

+++

Regression analysis is a set of statistical methodologies for determining the relationship between a dependent (or outcome) variable and one or more independent variables (also known as predictor variables). The most common algorithm used is Linear Regression. 

Linear Regression is a supervised learning technique that attempts to fit the best line that reduces the discrepancies between the predicted and the actual target value. This best fit line is arrived at by aiming to reduce the MSE (Mean Square Error).

+++

Let's import the required libraries and read the dataset.

```{code-cell} ipython3
import urllib.request
import pandas as pd 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn_evaluation import plot

# download dataset
urllib.request.urlretrieve('https://raw.githubusercontent.com/ploomber/sklearn-evaluation/master/docs-assets/regression/insurance.csv', filename='insurance.csv')

data = pd.read_csv('insurance.csv')
```

## Analyse the dataset

```{code-cell} ipython3
data.head()
```

Transform non-numerical labels to numerical labels.

```{code-cell} ipython3
#sex
le = LabelEncoder()
le.fit(data.sex.drop_duplicates()) 
data.sex = le.transform(data.sex)
# smoker or not
le.fit(data.smoker.drop_duplicates()) 
data.smoker = le.transform(data.smoker)
#region
le.fit(data.region.drop_duplicates()) 
data.region = le.transform(data.region)
```

Let's visualise the correlation among the variables.

```{code-cell} ipython3
sns.heatmap(data.corr(), annot=True,cmap='cool')
```

A strong correlation is observed with the smoking aspect of the patient. Now let's train a linear regression model on the data.

+++

## Train the model

```{code-cell} ipython3
X = data.drop(['charges'], axis = 1)
y = data.charges

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = LinearRegression().fit(X_train,y_train)
```

```{code-cell} ipython3
y_pred = model.predict(X_test)
print(r2_score(y_test, y_pred))
```

## Residual Plot

+++

R2 score is one measure for measuring how well the model fits the data. But we need other approaches as well to analyse the results. 

A residual is the error between the predicted value and the actual observed value (Observed - Predicted). A residual plot visualises the residuals on the Y-axis and the predicted values on the X-axis. A good residual plot should have the following characteristics:

- Points should cluster more towards the center of the plot.
- Density of points is more towards the origin as compared to away from the origin.
- There shouldn't be any clear pattern in the distribution of the points.

We can see that the below residual plot is fairly decent.

```{code-cell} ipython3
plot.residuals(y_test, y_pred)
```

## Prediction Error Plot

Another diagnostic tool to analyse the regression model is the prediction error plot. This plot visualises the observed target values against the values predicted by the model. Closer the best fit and the identity lines, better is the correlation between the predicted and the actual outcome.

```{code-cell} ipython3
plot.prediction_error(y_test, y_pred)
```

## Outlier Detection 

Cook's distance is a metric that can be used for detecting outliers in the dataset. It measures how much a regression model changes when a particular data point is removed from the dataset. From the plot we can see that the large spikes are the data points that have the capability to influence the model and affect the results significantly. Removal of these outlier points should be done after careful inspection.

```{code-cell} ipython3
plot.cooks_distance(X,y)
```
