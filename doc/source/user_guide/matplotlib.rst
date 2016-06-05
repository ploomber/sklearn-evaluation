Advanced usage using matplotlib
===============================

.. ipython:: python
    :suppress:

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import train_test_split
    from sklearn import datasets
    from sklearn_evaluation import plot

    import matplotlib.pyplot as plt

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    est = RandomForestClassifier()
    est.fit(X_train, y_train)

    y_true = y_test
    y_score = est.predict_proba(X_test)

    est2 = LogisticRegression()
    est2.fit(X_train, y_train)

    y_score2 = est2.predict_proba(X_test)

As we mentioned in the previous section, using the functional interface
provides great flexibility to evaluate your models, this sections includes
some recipes for common tasks that involve the use of the matplotlib API.

Changing plot style
*******************

sklearn-evaluation uses whatever configuration matplotlib has, if you want
to change the style of the plots easily you can use one of the many styles
available:

.. ipython:: python
    
    import matplotlib.style
    matplotlib.style.available

The change the style using

.. ipython:: python

    matplotlib.style.use('ggplot')


Let's see how a ROC curve looks with the new style:

.. ipython:: python
    
    @savefig roc_ggplot.png
    plot.roc(y_true, y_score)

Saving plots
************

.. ipython:: python
    
    ax = plot.roc(y_true, y_score)
    fig = ax.get_figure()
    fig.savefig('roc.png')

Comparing several models with one plot
**************************************

.. ipython:: python
    
    @savefig model_comparison.png
    plot.roc(y_true, y_score)
    plot.roc(y_true, y_score2)

Grid plots
**********

.. ipython:: python

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    plot.roc(y_true, y_score, ax=ax1)
    plot.roc(y_true, y_score2, ax=ax2)
    @savefig grid_plot.png
    plt.show()

Customizing plots
*****************

.. ipython:: python
    
    ax = plot.roc(y_true, y_score)
    ax.set_title('This is a custom title')
    @savefig roc_custom.png
    ax