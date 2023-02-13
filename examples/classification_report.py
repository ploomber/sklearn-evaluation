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

y_pred = est.predict(X_test)

# plot classification report for binary classification
plot.classification_report(y_test, y_pred, target_names=["Not spam", "Spam"])
