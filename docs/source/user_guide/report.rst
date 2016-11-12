Classifier evaluation using the OOP interface (report generation)
=================================================================

We can also use the :class:`sklearn_evaluation.ClassifierEvaluator` class to pack the results from our
estimator. This way we can generate plots and tables without having to pass
the parameters over and over again. If we are evaluating more than one model
at a time this also gives us a way to keep it organized. Furthermore, the ClassifierEvaluator class offers a way to create HTML reports
from our model results.

.. ipython:: python
    :suppress:

    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cross_validation import train_test_split
    from sklearn import datasets

    from sklearn_evaluation import ClassifierEvaluator

First, let's load some data and split for training and testing.

.. ipython:: python

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


Let's now train a classifier and predict on the test set.

.. ipython:: python

    est = RandomForestClassifier()
    est.fit(X_train, y_train)

    y_pred = est.predict(X_test)
    y_score = est.predict_proba(X_test)

    feature_list = range(4)
    target_names = ['setosa', 'versicolor', 'virginica']


Now that we have everything we need, let's pack our results using
ClassifierEvaluator, every parameter is optional.

.. ipython:: python

    ce = ClassifierEvaluator(est, y_test, y_pred, y_score,
                     feature_list, target_names,
                     estimator_name='RF')

We can use most of the functions in plot and table directly from the
ClassifierEvaluator object, let's see how to plot a confusion matrix.

.. ipython:: python

    @savefig cm_2.png
    ce.confusion_matrix

We can also generate HTML reports from our models by using the generate_report
function. The first parameter is a markdown template, we can either directly
pass a string or a path to a template.

We can include any attribute in the ClassifierEvaluator by passing a tag
with the format ``{attribute}``. Two extra tags are available ``{date}`` and
``{date_utc}`` which add local and UTC timestamps respectively.

.. ipython:: python

    template = '''
                # Report
                {estimator_type}
                {date}
                {confusion_matrix}
                {roc}
                {precision_recall}
               '''

    report = ce.generate_report(template)


The function will return an HTML string containing the report (images are
included in the same file). If we want to save it, we can use the path
parameter. Finally, if we want to apply custom CSS, we can use the style
parameter with a path to a CSS file.
