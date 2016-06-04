Report generation
=================

.. ipython:: python
    :suppress:

    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cross_validation import train_test_split
    from sklearn import datasets

    from sklearn_evaluation import ClassifierEvaluator


.. ipython:: python

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    est = RandomForestClassifier()
    est.fit(X_train, y_train)

    y_pred = est.predict(X_test)
    y_score = est.predict_proba(X_test)

    feature_list = range(4)
    target_names = ['setosa', 'versicolor', 'virginica']

    ce = ClassifierEvaluator(est, y_test, y_pred, y_score,
                         feature_list, target_names,
                         estimator_name='RF')

    template = '''
                # Report
                {estimator_type}
                {date}
                {confusion_matrix}
                {roc}
                {precision_recall}
               '''

    report = ce.generate_report(template)