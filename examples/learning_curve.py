from sklearn.model_selection import learning_curve
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

from sklearn_evaluation import plot

digits = load_digits()
X, y = digits.data, digits.target

# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = model_selection.ShuffleSplit(digits.data.shape[0],
                                  test_size=0.2, random_state=0)
cv = 5
estimator = GaussianNB()
train_sizes = np.linspace(.1, 1.0, 5)
train_sizes, train_scores, test_scores = learning_curve(
    estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_sizes)
plot.learning_curve(train_scores, test_scores, train_sizes)
plt.show()

# SVC is more expensive so we do a lower number of CV iterations:
cv = model_selection.ShuffleSplit(digits.data.shape[0],
                                  test_size=0.2, random_state=0)
cv = 5
estimator = SVC(gamma=0.001)
train_sizes = np.linspace(.1, 1.0, 5)
train_sizes, train_scores, test_scores = learning_curve(
    estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_sizes)

plot.learning_curve(train_scores, test_scores, train_sizes)
plt.show()
