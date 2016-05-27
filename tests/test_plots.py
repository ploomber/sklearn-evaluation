from unittest import TestCase
from sklearn_evaluation import plots
from sklearn.externals import joblib
from testing.image_testing import equal_images
import os

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

module_path = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(module_path, 'dummy_models')
result_path = os.path.join(module_path, 'result_images')

#Tolerance for image comparison
tol=50

#Tests missing:
#With bad input (wrong shape, not all classes present in non-binary form)


class Test_Confusion_Matrix(TestCase):
    def test_confusion_matrix(self):
        #Load y_pred, y_test
        y_pred = joblib.load(os.path.join(models_path,'confusion_matrix_y_pred.pkl'))
        y_test = joblib.load(os.path.join(models_path,'confusion_matrix_y_test.pkl'))
        #Generate plot
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        plots.confusion_matrix(y_test, y_pred,
                               target_names=['setosa', 'versicolor', 'virginica'],
                               ax=ax)
        #Save it
        fig.savefig(os.path.join(result_path, 'confusion_matrix.png'))
        #Compare
        result = equal_images(expected='baseline_images/confusion_matrix.png', actual='result_images/confusion_matrix.png', tol=tol, basepath=module_path)
        self.assertTrue(result)
    def test_normalized_confusion_matrix(self):
        #Load y_pred, y_test
        y_pred = joblib.load(os.path.join(models_path,'confusion_matrix_y_pred.pkl'))
        y_test = joblib.load(os.path.join(models_path,'confusion_matrix_y_test.pkl'))
        #Generate plot
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        plots.confusion_matrix(y_test, y_pred,
                ax=ax,
                target_names=['setosa', 'versicolor', 'virginica'],
                normalize=True)
        #Save it
        fig.savefig(os.path.join(result_path, 'normalized_confusion_matrix.png'))
        #Compare
        result = equal_images(expected='baseline_images/normalized_confusion_matrix.png', actual='result_images/normalized_confusion_matrix.png', tol=tol, basepath=module_path)
        self.assertTrue(result)

class Test_Feature_Importances(TestCase):
    def test_feature_importances(self):
        #Load model
        model = joblib.load(os.path.join(models_path,'feature_importances_model.pkl'))
        #Generate plot
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        plots.feature_importances(model, ax=ax)
        #Save it
        fig.savefig(os.path.join(result_path, 'feature_importances.png'))
        #Compare
        result = equal_images(expected='baseline_images/feature_importances.png', actual='result_images/feature_importances.png', tol=tol, basepath=module_path)
        self.assertTrue(result)

class Test_Precision_Recall(TestCase):
    def test_precision_recall(self):
        #Load y_score, y_test
        y_score = joblib.load(os.path.join(models_path,'precision_recall_y_score.pkl'))
        y_test = joblib.load(os.path.join(models_path,'precision_recall_y_test.pkl'))
        #Generate plot
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        plots.precision_recall(y_test, y_score, ax=ax)
        #Save plot
        fig.savefig(os.path.join(result_path, 'precision_recall.png'))
        #Compare
        result = equal_images(expected='baseline_images/precision_recall.png', actual='result_images/precision_recall.png', tol=tol, basepath=module_path)
        self.assertTrue(result)
    def test_multi_precision_recall(self):
        #Load y_score, y_test
        y_score = joblib.load(os.path.join(models_path,'multi_precision_recall_y_score.pkl'))
        y_test = joblib.load(os.path.join(models_path,'multi_precision_recall_y_test.pkl'))
        #Generate plot
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        plots.precision_recall(y_test, y_score, ax=ax)
        #Save plot
        fig.savefig(os.path.join(result_path, 'multi_precision_recall.png'))
        #Compare
        result = equal_images(expected='baseline_images/multi_precision_recall.png', actual='result_images/multi_precision_recall.png', tol=tol, basepath=module_path)
        self.assertTrue(result)

class Test_ROC(TestCase):
    def test_roc(self):
        #Load y_score, y_test
        y_score = joblib.load(os.path.join(models_path,'roc_y_score.pkl'))
        y_test = joblib.load(os.path.join(models_path,'roc_y_test.pkl'))
        #Generate plot
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        plots.roc(y_test, y_score, ax=ax)
        #Save plot
        fig.savefig(os.path.join(result_path, 'roc.png'))
        #Compare
        result = equal_images(expected='baseline_images/roc.png', actual='result_images/roc.png', tol=tol, basepath=module_path)
        self.assertTrue(result)
    def test_multi_roc(self):
        #Load y_score, y_test
        y_score = joblib.load(os.path.join(models_path,'multi_roc_y_score.pkl'))
        y_test = joblib.load(os.path.join(models_path,'multi_roc_y_test.pkl'))
        #Generate plot
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        plots.roc(y_test, y_score)
        #Save plot
        fig.savefig(os.path.join(result_path, 'multi_roc.png'))
        #Compare
        result = equal_images(expected='baseline_images/multi_roc.png', actual='result_images/multi_roc.png', tol=tol, basepath=module_path)
        self.assertTrue(result)