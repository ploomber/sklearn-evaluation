from sklearn_evaluation import ClassifierEvaluator

# import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def test_can_plot():
    data = datasets.make_classification(200, 10, n_informative=5, class_sep=0.65)
    X = data[0]
    y = data[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    est = RandomForestClassifier()
    est.fit(X_train, y_train)

    evaluator = ClassifierEvaluator(est, y_true=y_test, X=X_test)

    evaluator.confusion_matrix()
