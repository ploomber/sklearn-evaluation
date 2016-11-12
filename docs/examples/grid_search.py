import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import datasets

from sklearn_evaluation.plot import grid_search


iris = datasets.load_iris()


parameters = {
    'n_estimators': [1, 10, 50, 100],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2'],
}

est = RandomForestClassifier()
clf = GridSearchCV(est, parameters, cv=5)

#clf.fit(iris.data, iris.target)
data = datasets.make_classification(1000, 10, 5, class_sep=0.7)
clf.fit(data[0], data[1])

grid_scores = clf.grid_scores_

# changing numeric parameter without any restrictions
# in the rest of the parameter set
to_vary = 'n_estimators'
grid_search(clf.grid_scores_, to_vary)
plt.show()
# you can also use bars
grid_search(clf.grid_scores_, to_vary, kind='bar')
plt.show()

# changing a categorical variable
# without any constraints
to_vary = 'criterion'
grid_search(clf.grid_scores_, to_vary)
plt.show()
# bar
grid_search(clf.grid_scores_, to_vary, kind='bar')
plt.show()


# varying a numerical parameter but constraining
# the rest of the parameter set
to_vary = 'n_estimators'
to_keep = {'max_features': 'sqrt', 'criterion': 'gini'}
grid_search(clf.grid_scores_, to_vary, to_keep, kind='bar')
plt.show()
# same as above but letting max_features to have two values
to_keep = {'max_features': ['sqrt', 'log2'], 'criterion': 'gini'}
grid_search(clf.grid_scores_, to_vary, to_keep, kind='bar')
plt.show()

# varying two parameters - you can only show this as a
# matrix so the kind parameter will be ignored
to_vary = ('n_estimators', 'criterion')
to_keep = {'max_features': 'sqrt'}
grid_search(clf.grid_scores_, to_vary, to_keep)
plt.show()
