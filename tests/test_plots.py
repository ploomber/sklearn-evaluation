from unittest import TestCase
from matplotlib.testing.decorators import image_comparison
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sklearn_model_eval import plots
from sklearn.externals import joblib


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.image as mpimg


from matplotlib.testing.compare import compare_images

class TestDumb(TestCase):
     def test_true_is_true(self):
         self.assertTrue(True)

class Test_Confusion_Matrix(TestCase):
    def test_plot_is_correct(self):
        #Load y_pred, y_test
        y_pred = joblib.load('dummy_models/confusion_matrix_y_pred.pkl')
        y_test = joblib.load('dummy_models/confusion_matrix_y_test.pkl')
        #Generate plot
        cf = plots.confusion_matrix_(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica'])
        #Save it
        cf.savefig("result_images/test_plots/tmp.png")
        #Compare
        compare = compare_images(expected='baseline_images/test_plots/cf.png', actual='result_images/test_plots/tmp.png', tol=50, in_decorator=True)
        self.assertIsNone(compare)