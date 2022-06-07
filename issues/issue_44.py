# %%
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import datasets

sys.path.append('../src')
from sklearn_evaluation.plot import grid_search

std = StandardScaler()
est = KNeighborsClassifier()

pipe = Pipeline([
    ('std', std),
    ('est', est)
])

parameters = {
   'std': [None, std],
   'est__weights': ['distance', 'uniform'],
   'est__n_neighbors': [1, 10, 50, 100],
}

clf = GridSearchCV(pipe, parameters, cv=5)

X, y = datasets.make_classification(1000, 10, n_informative=5, class_sep=0.7)
clf.fit(X, y)

# changing numeric parameter without any restrictions
# in the rest of the parameter set
grid_search(clf.cv_results_, change='est__n_neighbors')
plt.show()

grid_search(clf.cv_results_, change='est__n_neighbors', subset={'std': None})
plt.show()

grid_search(clf.cv_results_, change=('est__weights', 'est__n_neighbors'), subset={'std': None})
plt.show()

# %% unsorted 
parameters = {
   'weights': ['uniform', 'distance'],
   'n_neighbors': [1, 10, 50, 100],
}

clf = GridSearchCV(est, parameters, cv=5)

clf.fit(X, y)

grid_search(clf.cv_results_, change='n_neighbors')
plt.show()

# %% sorted
parameters = {
   'weights': sorted(['uniform', 'distance']),
   'n_neighbors': [1, 10, 50, 100],
}

clf = GridSearchCV(est, parameters, cv=5)

clf.fit(X, y)

grid_search(clf.cv_results_, change='n_neighbors')
plt.show()

# %% labels
from functools import reduce

label = ['criterion: entropy', 'max_features: sqrt']
label_bad = reduce(lambda x, y: x + ', ' + y, label, '')
label_good = ', '.join(label)

print('label_bad =', label_bad)
print('label_good =', label_good)
