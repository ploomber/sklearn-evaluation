from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn_evaluation import plot

import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
plot.target_analysis(y_train, y_test)
plt.show()
