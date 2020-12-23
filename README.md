# sklearn-evaluation


![CI](https://github.com/edublancas/sklearn-evaluation/workflows/CI/badge.svg) [![Documentation Status](https://readthedocs.org/projects/sklearn-evaluation/badge/?version=latest)](https://sklearn-evaluation.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/sklearn-evaluation.svg)](https://badge.fury.io/py/sklearn-evaluation) [![Coverage Status](https://coveralls.io/repos/github/edublancas/sklearn-evaluation/badge.svg)](https://coveralls.io/github/edublancas/sklearn-evaluation)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edublancas/sklearn-evaluation/master?filepath=docs%2Fsource%2Fnbs)

Machine learning model evaluation made easy: plots, tables, HTML reports, experiment tracking and Jupyter notebook analysis.

Supports Python 3.6 and higher.

[Documentation here.](https://sklearn-evaluation.readthedocs.io)

![confusion matrix](examples/cm.png)

![grid search](https://sklearn-evaluation.readthedocs.io/en/stable/_images/gs_1.png)

# Install  

```bash
pip install sklearn-evaluation
```

# Features

* [Plotting](https://sklearn-evaluation.readthedocs.io/en/stable/_images/cm.png) (confusion matrix, feature importances, precision-recal, roc)
* Report generation ([example](http://htmlpreview.github.com/?https://github.com/edublancas/sklearn-model-evaluation/blob/master/examples/report.html))
* [Evaluate grid search results](https://sklearn-evaluation.readthedocs.io/en/stable/user_guide/grid_search.html)
* [Track experiments using a local SQLite database](https://sklearn-evaluation.readthedocs.io/en/stable/user_guide/SQLiteTracker.html)
* [Analyze notebooks output](https://sklearn-evaluation.readthedocs.io/en/stable/user_guide/NotebookCollection.html)