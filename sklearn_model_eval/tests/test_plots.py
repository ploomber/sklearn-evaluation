from unittest import TestCase
from matplotlib.testing.decorators import image_comparison
from sklearn_model_eval import plots
from sklearn.externals import joblib


# class TestDumb(TestCase):
#     def test_true_is_true(self):
#         self.assertTrue(True)

#@image_comparison(baseline_images=['test.jpg'])
def test_plot_is_correct():
    #Load y_pred, y_test
    y_pred = joblib.load('tests/dummy_models/confusion_matrix_y_pred.pkl')
    y_test = joblib.load('tests/dummy_models/confusion_matrix_y_test.pkl')
    #Generate plot
    cf = plots.confusion_matrix_(y_test, y_pred, target_names=['A', 'B'])
    #Check plot is correct
    print cf
    cf