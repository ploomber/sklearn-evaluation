Evaluating Grid Search Results
==============================

.. ipython:: python
    :suppress:

    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.grid_search import GridSearchCV
    from sklearn import datasets

    from sklearn_evaluation.plot import grid_search


.. ipython:: python

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    parameters = {
        'n_estimators': [1, 10, 50, 100],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
    }
    
    est = RandomForestClassifier()
    clf = GridSearchCV(est, parameters, cv=3)
    clf.fit(X, y)
    grid_scores = clf.grid_scores_


.. ipython:: python

    @savefig gs_1.png
    grid_search(clf.grid_scores_, change='n_estimators', kind='bar')


.. ipython:: python

    @savefig gs_2.png
    grid_search(clf.grid_scores_, change=('n_estimators', 'criterion'), subset={'max_features': 'sqrt'}, kind='bar')