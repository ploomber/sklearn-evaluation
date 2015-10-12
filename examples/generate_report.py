from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split

from sklearn_model_eval.report import generate_report

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)

y_score = y_pred
feature_list = range(4)
target_names = ['setosa', 'versicolor', 'virginica']

generate_report(classifier, y_test, y_pred, y_score, feature_list, target_names, path='/Users/Edu/Desktop/', name='test.html')