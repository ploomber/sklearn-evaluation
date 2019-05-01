from jinja2 import Environment, DictLoader


from sklearn_evaluation import ClassifierEvaluator

import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


data = datasets.make_classification(200, 10, 5, class_sep=0.65)
X = data[0]
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

est = RandomForestClassifier()
est.fit(X_train, y_train)

evaluator = ClassifierEvaluator(est, y_true=y_test, X=X_test)

from sklearn_evaluation.report import figure2html


class EvaluatorHTMLSerializer:
    """Serializes elements in an evaluator to an appropriate format
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def __getattr__(self, key):
        attr = getattr(self.evaluator, key)

        if callable(attr):
            return HTMLSerializableCallable(attr)
        else:
            return attr


class HTMLSerializableCallable:

    def __init__(self, attr):
        self.attr = attr

    def __call__(self, *args, **kwargs):
        obj = self.attr(*args, **kwargs)

        if isinstance(obj, matplotlib.axes.Axes):
            return figure2html(obj.get_figure())
        elif isinstance(obj, matplotlib.figure.Figure):
            return figure2html(obj)
        else:
            raise TypeError('Unsupported type {}'.format(type(obj)))


serializer = EvaluatorHTMLSerializer(evaluator)
serializer.confusion_matrix()


ax = evaluator.confusion_matrix()
plt.show()

template = """ {{evaluator.confusion_matrix()}} """

env = Environment(
    loader=DictLoader({'template': template})
)

t = env.get_template('template')

t.render(evaluator=EvaluatorHTMLSerializer(evaluator))
