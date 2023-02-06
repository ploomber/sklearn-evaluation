from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn_evaluation import plot
import matplotlib.pyplot as plt

# load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

# plot residuals
plot.residuals(y_test, y_pred)
