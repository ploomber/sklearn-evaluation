from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn_evaluation.plot import confusion_matrix


from matplotlib import style

style.use("seaborn-dark")

# Import some data to play with
X, y = datasets.make_classification(
    1000, 10, n_informative=5, class_sep=0.7, n_classes=8
)

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

est = RandomForestClassifier()
est.fit(X_train, y_train)
y_pred = est.predict(X_test)
y_true = y_test

# plot normalized confusion matrix
confusion_matrix(y_true, y_pred, normalize=True)

# plot non-normalized confusion matrix
confusion_matrix(y_true, y_pred)
