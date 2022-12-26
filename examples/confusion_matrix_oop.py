from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn_evaluation import plot

X, y = datasets.make_classification(
    1000, 20, n_informative=10, class_sep=0.80, n_classes=3, random_state=0
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

est = RandomForestClassifier()
est.fit(X_train, y_train)
y_pred = est.predict(X_test)

tree_cm = plot.ConfusionMatrix.from_raw_data(y_test, y_pred)

est = DecisionTreeClassifier()
est.fit(X_train, y_train)
y_pred = est.predict(X_test)
forest_cm = plot.ConfusionMatrix(y_test, y_pred)

# Compare
tree_cm + forest_cm

# Diff
forest_cm - tree_cm
