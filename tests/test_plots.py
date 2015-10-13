from unittest import TestCase
from sklearn_model_eval import plots
from sklearn.externals import joblib
from testing.image_testing import equal_images
import os

module_path = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(module_path, 'dummy_models')
result_path = os.path.join(module_path, 'result_images')

class Test_Confusion_Matrix(TestCase):
    def test_confusion_matrix(self):
        #Load y_pred, y_test
        y_pred = joblib.load(os.path.join(models_path,'confusion_matrix_y_pred.pkl'))
        y_test = joblib.load(os.path.join(models_path,'confusion_matrix_y_test.pkl'))
        #Generate plot
        cf = plots.confusion_matrix_(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica'])
        #Save it
        cf.savefig(os.path.join(result_path, 'confusion_matrix.png'))
        #Compare
        result = equal_images(expected='baseline_images/confusion_matrix.png', actual='result_images/confusion_matrix.png', tol=38, basepath=module_path)
        self.assertTrue(result)
    def test_normalized_confusion_matrix(self):
        #Load y_pred, y_test
        y_pred = joblib.load(os.path.join(models_path,'confusion_matrix_y_pred.pkl'))
        y_test = joblib.load(os.path.join(models_path,'confusion_matrix_y_test.pkl'))
        #Generate plot
        ncf = plots.confusion_matrix_(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica'], normalize=True, title="Normalized confusion matrix")
        #Save it
        ncf.savefig(os.path.join(result_path, 'normalized_confusion_matrix.png'))
        #Compare
        result = equal_images(expected='baseline_images/normalized_confusion_matrix.png', actual='result_images/normalized_confusion_matrix.png', tol=50, basepath=module_path)
        self.assertTrue(result)