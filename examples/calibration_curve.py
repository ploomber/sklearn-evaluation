import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn_evaluation import plot

# generate data
X, y = make_classification(
    n_samples=20000, n_features=2, n_informative=2, n_redundant=0, random_state=0
)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0
)

rf = RandomForestClassifier()
lr = LogisticRegression()
nb = GaussianNB()

rf_probas = rf.fit(X_train, y_train).predict_proba(X_test)
lr_probas = lr.fit(X_train, y_train).predict_proba(X_test)
nb_probas = nb.fit(X_train, y_train).predict_proba(X_test)

# list of probabilities for different classifier
probabilities = [rf_probas, lr_probas, nb_probas]

clf_names = [
    "Random Forest",
    "Logistic Regression",
    "Gaussian Naive Bayes",
]

# plot calibration curve
plot.calibration_curve(y_test, probabilities, clf_names)
plt.show()
