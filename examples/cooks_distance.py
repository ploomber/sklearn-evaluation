import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

from sklearn_evaluation import plot


X, y_true = fetch_california_housing(return_X_y=True)

reg = LinearRegression().fit(X, y_true)

y_pred = reg.predict(X)

plot.cooks_distance(X, y_true, y_pred)
plt.show()
