import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

from sklearn_evaluation import plot

housing = datasets.fetch_california_housing()
X, y_true = housing['data'], housing['target']

reg = LinearRegression().fit(X, y_true)

y_pred = reg.predict(X)

plot.cooks_distance(X, y_true, y_pred)
plt.show()