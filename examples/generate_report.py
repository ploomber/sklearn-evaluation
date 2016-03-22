import numpy as np
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

from sklearn_evaluation.model_results import ClassificationModelResults
from sklearn_evaluation.report import ReportGenerator

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
classifier = classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_score = classifier.decision_function(X_test)

feature_list = range(4)
target_names = ['setosa', 'versicolor', 'virginica']

#Create a trained model instance
tm = ClassificationModelResults(classifier, y_test, y_pred, y_score,
    feature_list, target_names, model_name='sample_model_report')

#Generate report
report_gen = ReportGenerator(savepath='/Users/Edu/Desktop')
report_gen(tm)