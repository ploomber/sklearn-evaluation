from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn_evaluation import plot

data = datasets.make_classification(200, 10, n_informative=5, class_sep=0.65)
X = data[0]
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

est = RandomForestClassifier()
est.fit(X_train, y_train)

y_pred = est.predict(X_test)
y_score = est.predict_proba(X_test)
y_true = y_test

cm1 = plot.ConfusionMatrix(y_true, y_score)

# create another dataset
data_ = datasets.make_classification(200, 10, n_informative=5, class_sep=0.15)
X_ = data_[0]
y_ = data_[1]

X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.3)

est_ = RandomForestClassifier()
est_.fit(X_train_, y_train_)

y_pred_ = est.predict(X_test_)
y_score_ = est.predict_proba(X_test_)
y_true_ = y_test_

cm2 = plot.ROC(y_true_, y_score_)

# Generate a new plot with overlapping curves
cm1 + cm2
