from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn_evaluation import plot

# generate data
X, y = datasets.make_classification(200, 10, n_informative=5, class_sep=0.65)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

est = RandomForestClassifier()
est.fit(X_train, y_train)

y_score = est.predict_proba(X_test)
y_true = y_test

roc = plot.ROC.from_raw_data(y_true, y_score)

# create another dataset
X_, y_ = datasets.make_classification(200, 10, n_informative=5, class_sep=0.15)

# split data into train and test
X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.3)

est_ = RandomForestClassifier()
est_.fit(X_train_, y_train_)

y_score_ = est.predict_proba(X_test_)
y_true_ = y_test_

roc2 = plot.ROC.from_raw_data(y_true_, y_score_)

# Compare both classifiers
roc + roc2
