from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn_evaluation.plot import confusion_matrix

import matplotlib.pyplot as plt

from matplotlib import style
style.use('seaborn-dark')

# Import some data to play with
data = datasets.make_classification(1000, 10, 5, class_sep=0.7, n_classes=8)
X = data[0]
y = data[1]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)


est = RandomForestClassifier()
est.fit(X_train, y_train)
y_pred = est.predict(X_test)
y_true = y_test

confusion_matrix(y_true, y_pred, normalize=True)
plt.show()

confusion_matrix(y_true, y_pred)
plt.show()
