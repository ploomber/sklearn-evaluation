from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn_evaluation import plot

# generate data
X, y = datasets.make_classification(200, 10, n_informative=5, class_sep=0.65)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

y_pred_rf = RandomForestClassifier().fit(X_train, y_train).predict(X_test)
y_pred_lr = LogisticRegression().fit(X_train, y_train).predict(X_test)

target_names = ["Not spam", "Spam"]

# report for random forest
cr_rf = plot.ClassificationReport.from_raw_data(
    y_test, y_pred_rf, target_names=target_names
)

# report for logistic regression
cr_lr = plot.ClassificationReport.from_raw_data(
    y_test, y_pred_lr, target_names=target_names
)

# how better it is the random forest?
cr_rf - cr_lr

# compare both reports
cr_rf + cr_lr
