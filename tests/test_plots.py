from unittest import TestCase
from sklearn_model_eval import plots
from sklearn.externals import joblib
from testing.image_testing import equal_images


class Test_Confusion_Matrix(TestCase):
    def test_confusion_matrix(self):
        #Load y_pred, y_test
        y_pred = joblib.load('tests/dummy_models/confusion_matrix_y_pred.pkl')
        y_test = joblib.load('tests/dummy_models/confusion_matrix_y_test.pkl')
        #Generate plot
        cf = plots.confusion_matrix_(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica'])
        #Save it
        cf.savefig("tests/result_images/test_plots/cf.png")
        #Compare
        result = equal_images(expected='tests/baseline_images/test_plots/cf.png', actual='tests/result_images/test_plots/cf.png')
        self.assertTrue(result)
    def test_normalized_confusion_matrix(self):
        #Load y_pred, y_test
        y_pred = joblib.load('tests/dummy_models/confusion_matrix_y_pred.pkl')
        y_test = joblib.load('tests/dummy_models/confusion_matrix_y_test.pkl')
        #Generate plot
        ncf = plots.confusion_matrix_(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica'], normalize=True, title="Normalized confusion matrix")
        #Save it
        ncf.savefig("tests/result_images/test_plots/ncf.png")
        #Compare
        result = equal_images(expected='tests/baseline_images/test_plots/ncf.png', actual='tests/result_images/test_plots/ncf.png')
        self.assertTrue(result)