import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn_evaluation import plot

# generate data
X, y = make_classification(
    n_samples=10000, n_features=2, n_informative=2, n_redundant=0, random_state=0
)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0
)

clf = LogisticRegression()

y_scores = clf.fit(X_train, y_train).predict_proba(X_test)

# plot scores distribution
plot.scores_distribution(y_scores[:, 1], n_bins=10)
