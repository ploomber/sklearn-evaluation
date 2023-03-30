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
myst:
  html_meta:
    description lang=en: "Learn experiment tracking with SQLiteTracker for comparing and analyzing Machine Learning models."
    keywords: "SQLiteTracker, experiment tracking, machine learning, model comparison, sklearn_evaluation, RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, accuracy_score"
    property=og:locale: "en_US"
notebook_metadata_filter: myst
---

# Experiment tracking

 `SQLiteTracker` provides a powerful and flexible way to track computational (e.g., Machine Learning) experiments using a SQLite database. Allows you to use SQL as the query language, giving you a powerful tool for experiment comparison, and comes with plotting features to compare plots side-by-side and to combine plots for better comparison.

Read more about the motivations in our [blog post](https://ploomber.io/blog/experiment-tracking/), check out the [HN discussion.](https://news.ycombinator.com/item?id=33624018)

This tutorial will walk you through the features with a Machine Learning use case; however, the tracker is generic enough to be used in any other domains.

```{code-cell} ipython3
from pathlib import Path

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

```{code-cell} ipython3
# delete our example database, if any
db = Path("my_experiments.db")

if db.exists():
    db.unlink()
```

```{code-cell} ipython3
:tags: [parameters]

from sklearn_evaluation import SQLiteTracker

tracker = SQLiteTracker("my_experiments.db")
```

```{code-cell} ipython3
X, y = datasets.make_classification(200, 10, n_informative=5, class_sep=0.65)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

models = [RandomForestClassifier(), LogisticRegression(), DecisionTreeClassifier()]
```

## Training and logging models

```{code-cell} ipython3
:tags: [hide-output]

for m in models:
    model = type(m).__name__
    print(f"Fitting {model}")

    experiment = tracker.new_experiment()
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # log a dictionary with log_dict
    experiment.log_dict({"accuracy": acc, "model": model, **m.get_params()})
```


## Displaying latest experiments

Display the `tracker` object to show last experiments:

```{code-cell} ipython3
tracker
```

```{eval-rst}
.. tip::
   
   :doc:`Click here <../comparison/SQLiteTracker>` to see the detailed user guide.
```