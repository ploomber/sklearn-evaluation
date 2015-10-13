from unittest import TestCase
from sklearn_model_eval.report import ReportGenerator

class Test_Complete_Classification_Report(TestCase):
    def test_default_template(self):
        rg = ReportGenerator(savepath='')
        rg.generate_report(model='', y_true='', y_pred='', y_score='', feature_list='', target_names='', name='')
        pass
    def test_default_template_without_style(self):
        rg = ReportGenerator(savepath='', css=None)
        pass
    def test_custom_template(self):
        pass

class Test_Partial_Classification_Report(TestCase):
    pass