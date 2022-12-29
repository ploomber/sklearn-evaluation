import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import datasets

from sklearn_evaluation.plot import grid_search


iris = datasets.load_iris()


parameters = {
    "n_estimators": [1, 10, 50, 100],
    "criterion": ["gini", "entropy"],
    "max_features": ["sqrt", "log2"],
}

est = RandomForestClassifier()
clf = GridSearchCV(est, parameters, cv=5)

X, y = datasets.make_classification(1000, 10, n_informative=5, class_sep=0.7)
clf.fit(X, y)

# changing numeric parameter without any restrictions
# in the rest of the parameter set
grid_search(clf.cv_results_, change="n_estimators")
plt.show()

# you can also use bars
grid_search(clf.cv_results_, change="n_estimators", kind="bar")
plt.show()

# changing a categorical variable without any constraints
grid_search(clf.cv_results_, change="criterion")
plt.show()

# bar
grid_search(clf.cv_results_, change="criterion", kind="bar")
plt.show()


# varying a numerical parameter but constraining
# the rest of the parameter set
grid_search(
    clf.cv_results_,
    change="n_estimators",
    subset={"max_features": "sqrt", "criterion": "gini"},
    kind="bar",
)
plt.show()

# same as above but letting max_features to have two values
grid_search(
    clf.cv_results_,
    change="n_estimators",
    subset={"max_features": ["sqrt", "log2"], "criterion": "gini"},
    kind="bar",
)
plt.show()

# varying two parameters - you can only show this as a
# matrix so the kind parameter will be ignored
grid_search(
    clf.cv_results_,
    change=("n_estimators", "criterion"),
    subset={"max_features": "sqrt"},
)
plt.show()
