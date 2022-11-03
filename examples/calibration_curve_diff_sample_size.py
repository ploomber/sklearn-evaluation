import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn_evaluation import plot


def make_dataset(n_samples):
    X, y = make_classification(n_samples=n_samples,
                               n_features=2,
                               n_informative=2,
                               n_redundant=0,
                               random_state=0)
    return train_test_split(X, y, test_size=0.33, random_state=0)


X_train, X_test, y_train, y_test1 = make_dataset(n_samples=1000)
probs1 = LogisticRegression().fit(X_train, y_train).predict_proba(X_test)

X_train, X_test, y_train, y_test2 = make_dataset(n_samples=10000)
probs2 = LogisticRegression().fit(X_train, y_train).predict_proba(X_test)

# if you want plot probability curves for different sample sizes, pass
# a list with the true labels per each element in the probabilities
# argyment
plot.calibration_curve([y_test1, y_test2], [probs1, probs2],
                       clf_names=['1k samples', '10k samples'])
plt.show()
