import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn_evaluation import plot
from sklearn import datasets

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
plot.target_plot(y_train)
plt.show()

kwargs = {
        "n_samples": 100,
        "n_classes": 5,
        "n_features": 20,
        "n_informative": 8,
        "n_redundant": 2,
        "n_clusters_per_class": 1,
        "random_state": 89092,
    }

X, y = datasets.make_classification(**kwargs)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

plot.target_plot(y_train, y_test, mode='compare')
plt.show()
