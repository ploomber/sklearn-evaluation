"""
Feature importances plot

%load_ext autoreload
%autoreload 2
"""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn_evaluation import plot

X, y = datasets.make_classification(200, 20, 5, class_sep=0.65)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier(n_estimators=1)
model.fit(X_train, y_train)

# plot all features
plot.feature_importances(model)
plt.show()

# only top 5
plot.feature_importances(model, top_n=5)
plt.show()
