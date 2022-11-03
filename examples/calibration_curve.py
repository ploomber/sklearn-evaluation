import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from sklearn_evaluation import plot

X, y = make_classification(n_samples=20000,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=0)

rf = RandomForestClassifier()
lr = LogisticRegression()
nb = GaussianNB()
svm = LinearSVC()

rf_probas = rf.fit(X_train, y_train).predict_proba(X_test)
lr_probas = lr.fit(X_train, y_train).predict_proba(X_test)
nb_probas = nb.fit(X_train, y_train).predict_proba(X_test)
svm_scores = svm.fit(X_train, y_train).decision_function(X_test)

probabilities = [rf_probas, lr_probas, nb_probas, svm_scores]

clf_names = [
    'Random Forest',
    'Logistic Regression',
    'Gaussian Naive Bayes',
    'Support Vector Machine',
]

plot.calibration_curve(y_test, probabilities, clf_names)
plt.show()
