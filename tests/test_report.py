from unittest import TestCase
from sklearn_model_eval.report import ReportGenerator
from sklearn.externals import joblib

module_path = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(module_path, 'dummy_models')

class Test_Complete_Classification_Report(TestCase):
    def test_default_template(self):
        #Load a model
        model = joblib.load(os.path.join(models_path,'feature_importances_model.pkl'))
        #Load y_true
        y_true = None
        #Load y_pred
        y_pred = None
        #Load y_score
        y_score = None
        #Generate feature list
        feature_list = None
        #Generate target names
        target_names = None
        #Report name
        name = 'test_report'
        rg = ReportGenerator()
        report = rg.generate_report(model=model, y_true=y_true, y_pred=y_pred, y_score=y_score, feature_list='', target_names='', name='')
        #Compare report
        pass
    def test_default_template_model_without_feature_importances(self):
        pass
    def test_default_template_without_style(self):
        rg = ReportGenerator(css=None)
        pass
    def test_custom_template(self):
        pass
    def test_report_saved_to_disk(self):
        rg = ReportGenerator(savepath='')
        pass

class Test_Partial_Classification_Report(TestCase):
    pass