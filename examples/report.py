from pathlib import Path

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn_evaluation import ClassifierEvaluator

X, y = datasets.make_classification(200, 10, 5, class_sep=0.65)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier(n_estimators=5).fit(X_train, y_train)

evaluator = ClassifierEvaluator(model, y_true=y_test, X=X_test)
report = evaluator.make_report()

p = Path('~/Desktop/report.html').expanduser()
report.save(p)
