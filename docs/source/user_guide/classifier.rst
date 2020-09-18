Classifier evaluation
=====================

.. ipython:: python
    :suppress:

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn_evaluation import plot, table

    import matplotlib.pyplot as plt

sklearn-evluation has two main modules for evaluating classifiers: :mod:`sklearn_evaluation.plot` and
:mod:`sklearn_evaluation.table`, let's see an example of how to use them.

First, let's load some data and split it in training and test set.

.. ipython:: python

    data = datasets.make_classification(200, 10, n_informative=5,
                                        class_sep=0.65)
    X = data[0]
    y = data[1]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


Now, we are going to train the data using one of the scikit-learn classifiers.

.. ipython:: python

    est = RandomForestClassifier(n_estimators=5)
    est.fit(X_train, y_train)

Most of the functions require us to pass the class predictions for the test
set (``y_pred``), the scores assigned (``y_score``) and the ground truth classes
(``y_true``), let's define such variables.

.. ipython:: python

    y_pred = est.predict(X_test)
    y_score = est.predict_proba(X_test)
    y_true = y_test

We can start evaluating our model, the following example shows how to plot
a confusion matrix.

.. ipython:: python

    @savefig cm.png
    plot.confusion_matrix(y_true, y_pred)


Some classifiers (such as :class:`sklearn.ensemble.RandomForestClassifier`) have feature importances, we can plot
then passing the estimator object to the feature_importances function.

.. ipython:: python

    @savefig fi.png
    plot.feature_importances(est, top_n=5)

A feature importances function is also available in the table module.

.. ipython:: python

    print(table.feature_importances(est))

Now, let's see how to generate two of the most common plots for evaluating
classifiers: Precision-Recall and ROC.

.. ipython:: python

    @savefig pr.png
    plot.precision_recall(y_true, y_score)


.. ipython:: python

    @savefig roc.png
    plot.roc(y_true, y_score)