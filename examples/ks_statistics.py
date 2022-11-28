import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn_evaluation.plot import plot_ks_statistic

X, y = make_classification(n_samples=1000,
                           n_features=20,
                           n_informative=10,
                           class_sep=0.80,
                           n_classes=2,
                           random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=123)
model = LogisticRegression(random_state=101)

model = model.fit(X_train, y_train)
y_probas = model.predict_proba(X_test)
plot_ks_statistic(y_test, y_probas)
plt.show()
