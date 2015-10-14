from sklearn import datasets
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn_model_eval.report import ReportGenerator

#To pickle y_test and y_pred
from sklearn.externals import joblib
#To work with paths
import os

#Paths for file saving
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
models_path = os.path.join(module_path, 'dummy_models')
baselline_path = os.path.join(module_path, 'baseline_images')

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Train
svc = SVC(probability=True)
svc.fit(X_train, y_train)

y_true = y_test
y_pred = svc.predict(X_test)
y_score = svc.predict_proba(X_test)

#Pickle model
joblib.dump(svc, os.path.join(models_path, 'classifier_without_feature_importances_model.pkl'))
#Pickle y_true
joblib.dump(y_true, os.path.join(models_path, 'classifier_without_feature_importances_y_true.pkl'))
#Pickle y_pred
joblib.dump(y_pred, os.path.join(models_path, 'classifier_without_feature_importances_y_pred.pkl'))
#Pickle y_score
joblib.dump(y_score, os.path.join(models_path, 'classifier_without_feature_importances_y_score.pkl'))
#Pickle X
joblib.dump(X, os.path.join(models_path, 'classifier_without_feature_importances_x.pkl'))

#Generate feature list
feature_list = range(X.shape[1])
#Generate target names
target_names = range(len(set(y)))
#Name
name = 'test_report_without_feature_importances'

#Generate report
rg = ReportGenerator()
report = rg.generate_report(model=svc, y_true=y_true, y_pred=y_pred, y_score=y_score, feature_list=feature_list, target_names=target_names, name=name)
#Save report to file
report_file = open(os.path.join(models_path, 'classiffier_without_feature_importances_report.html'), 'w')
report_file.write(report)
report_file.close()