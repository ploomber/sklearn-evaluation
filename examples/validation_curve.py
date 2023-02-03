import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve

from sklearn_evaluation import plot

# load data
digits = load_digits()
X, y = digits.data, digits.target

param_range = np.logspace(-6, -1, 5)
param_name = "gamma"
train_scores, test_scores = validation_curve(
    SVC(),
    X,
    y,
    param_name=param_name,
    param_range=param_range,
    cv=5,
    scoring="accuracy",
    n_jobs=1,
)

# plot validation curve for "gamma" hyperparameter
plot.validation_curve(train_scores, test_scores, param_range, param_name, semilogx=True)
plt.show()

param_range = np.array([1, 10, 100])
param_name = "n_estimators"
train_scores, test_scores = validation_curve(
    RandomForestClassifier(),
    X,
    y,
    param_name=param_name,
    param_range=param_range,
    cv=10,
    scoring="accuracy",
    n_jobs=1,
)

# plot validation curve for "n_estimators" hyperparameter
plot.validation_curve(
    train_scores, test_scores, param_range, param_name, semilogx=False
)
plt.show()
