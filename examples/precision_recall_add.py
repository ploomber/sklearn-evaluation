from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn_evaluation import plot

# generate data
X, y = datasets.make_classification(
    n_samples=200, n_features=10, n_informative=5, class_sep=0.65
)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

est = RandomForestClassifier()
est.fit(X_train, y_train)

y_pred = est.predict(X_test)
y_score = est.predict_proba(X_test)
y_true = y_test

# precision recall plot for random forest
forest_pr = plot.PrecisionRecall.from_raw_data(
    y_true, y_score, label="Random forest classifier"
)

est = DecisionTreeClassifier()
est.fit(X_train, y_train)
y_pred = est.predict(X_test)
y_score = est.predict_proba(X_test)
y_true = y_test

# precision recall plot for decision tree
tree_pr = plot.PrecisionRecall.from_raw_data(
    y_true, y_score, label="Decision Tree classifier"
)

# compare two precision recall curves
forest_pr + tree_pr
