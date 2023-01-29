import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn_evaluation import plot

data = datasets.make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_classes=5,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=0,
)
X = data[0]
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

est = RandomForestClassifier()
est.fit(X_train, y_train)

y_pred = est.predict(X_test)
y_score = est.predict_proba(X_test)
y_true = y_test

plot.precision_recall(y_true, y_score)
plt.show()
