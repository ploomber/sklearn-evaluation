Evaluating a classifier
=======================

.. ipython:: python
    :suppress:

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cross_validation import train_test_split
    from sklearn import datasets
    from sklearn_evaluation import plot

    import matplotlib.pyplot as plt


.. ipython:: python

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)
    est = RandomForestClassifier()
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    y_true = y_test
    y_score = est.predict_proba(X_test)

    @savefig cm.png
    plot.confusion_matrix(y_true, y_pred)


.. ipython:: python

    @savefig fi.png
    plot.feature_importances(est)


.. ipython:: python

    @savefig pr.png
    plot.precision_recall(y_true, y_score)


.. ipython:: python

    @savefig roc.png
    plot.roc(y_true, y_score)