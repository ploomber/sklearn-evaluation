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
html_meta:
  "description lang=en": "Learn how to create an interactive confusion matrix using the penguins dataset, with the help of sklearn and seaborn libraries in Python."
  "keywords": "interactive confusion matrix, penguins dataset, sklearn, seaborn, Python, machine learning, model evaluation, DecisionTreeClassifier"
  "property=og:locale": "en_US"
---

# Interactive Confusion Matrix

In this tutorial, we'll demonstrate how to plot an interactive confusion matrix using the `penguins` dataset.

```{code-cell} ipython3
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
```

 
## Load the dataset

```{code-cell} ipython3
df = sns.load_dataset("penguins")
df.dropna(inplace=True)
Y = df.species
Y = Y.map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})
df.drop("species", inplace=True, axis=1)
se = pd.get_dummies(df["sex"], drop_first=True)
df = pd.concat([df, se], axis=1)
df.drop("sex", axis=1, inplace=True)
le = LabelEncoder()
df["island"] = le.fit_transform(df["island"])
X = df
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=40
)
df.head()
```

## Train a model

```{code-cell} ipython3
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Interactive confusion matrix

```{code-cell} ipython3
from sklearn_evaluation import plot

cm = plot.InteractiveConfusionMatrix.from_raw_data(
    y_test.tolist(),
    y_pred.tolist(),
    X_test=X_test,
    feature_subset=[
        "Male",
        "body_mass_g",
        "bill_depth_mm",
        "bill_length_mm",
        "flipper_length_mm",
    ],
    nsample=6,
)
```

Clicking on each quadrant displays two tables: `Sample Observations` and `Quadrant Statistics`.
`Sample Observations` displays 5 random samples from the quadrant. `Quadrant Statistics` table displays some statistics on all the data that lies in this quadrant.

```{code-cell} ipython3
cm.chart
```
