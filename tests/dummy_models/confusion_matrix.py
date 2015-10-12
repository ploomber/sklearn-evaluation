#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

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

#Pickle arrays so tests can use it
joblib.dump(y_test, 'confusion_matrix_y_test.pkl')
joblib.dump(y_pred, 'confusion_matrix_y_pred.pkl')