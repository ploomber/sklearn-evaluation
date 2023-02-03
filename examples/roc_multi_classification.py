import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn_evaluation import plot

# load data
iris = load_iris()
X, y = iris.data, iris.target
y = iris.target_names[y]

random_state = np.random.RandomState(0)
n_samples, n_features = X.shape

X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)
(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

classifier = LogisticRegression()
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

# plot roc curve
roc = plot.ROC.from_raw_data(y_test, y_score)
roc
