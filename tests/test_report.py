from unittest import TestCase
from sklearn_model_eval.report import ReportGenerator
from sklearn.externals import joblib
from freezegun import freeze_time

import os

module_path = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(module_path, 'dummy_models')

@freeze_time("2015-10-14 10:42") #Date when the dummy report was created
class Test_Complete_Classification_Report(TestCase):
    def test_default_template(self):
        #Load a model
        model = joblib.load(os.path.join(models_path,'classifier_with_feature_importances_model.pkl'))
        #Load y_true
        y_true = joblib.load(os.path.join(models_path,'classifier_with_feature_importances_y_true.pkl'))
        #Load y_pred
        y_pred = joblib.load(os.path.join(models_path,'classifier_with_feature_importances_y_pred.pkl'))
        #Load y_score
        y_score = joblib.load(os.path.join(models_path,'classifier_with_feature_importances_y_score.pkl'))
        #Load X
        X = joblib.load(os.path.join(models_path,'classifier_with_feature_importances_x.pkl'))
        #Generate feature list
        feature_list = range(X.shape[1]) #FIND A BETTER WAY TO DO THIS
        #Generate target names
        target_names = range(len(set(y_true))) #FIND A BETTER WAY
        #Report name
        name = 'test_report'
        rg = ReportGenerator()
        report = rg.generate_report(model=model, y_true=y_true, y_pred=y_pred, y_score=y_score, feature_list=feature_list, target_names=target_names, name=name)
        #Load dummy report
        dummy_report_file = open(os.path.join(models_path, 'classiffier_with_feature_importances_report.html'))
        dummy_report = dummy_report_file.read()
        #Compare report
        self.assertEqual(report, dummy_report)
    def test_default_template_no_feature_importances(self):
        #Load a model
        model = joblib.load(os.path.join(models_path,'classifier_without_feature_importances_model.pkl'))
        #Load y_true
        y_true = joblib.load(os.path.join(models_path,'classifier_without_feature_importances_y_true.pkl'))
        #Load y_pred
        y_pred = joblib.load(os.path.join(models_path,'classifier_without_feature_importances_y_pred.pkl'))
        #Load y_score
        y_score = joblib.load(os.path.join(models_path,'classifier_without_feature_importances_y_score.pkl'))
        #Load X
        X = joblib.load(os.path.join(models_path,'classifier_without_feature_importances_x.pkl'))
        #Generate feature list
        feature_list = range(X.shape[1]) #FIND A BETTER WAY TO DO THIS
        #Generate target names
        target_names = range(len(set(y_true))) #FIND A BETTER WAY
        #Report name
        name = 'test_report_without_feature_importances'
        rg = ReportGenerator()
        report = rg.generate_report(model=model, y_true=y_true, y_pred=y_pred, y_score=y_score, feature_list=feature_list, target_names=target_names, name=name)
        #Load dummy report
        dummy_report_file = open(os.path.join(models_path, 'classiffier_without_feature_importances_report.html'))
        dummy_report = dummy_report_file.read()
        #Compare report
        self.assertEqual(report, dummy_report)
    def test_default_template_no_model(self):
        pass
    def test_default_template_no_scores(self):
        pass
    def test_default_template_no_target_names(self):
        pass
    def test_default_template_model_without_feature_importances(self):
        pass
    def test_default_template_without_style(self):
        #rg = ReportGenerator(css=None)
        pass
    def test_custom_template(self):
        pass
    def test_report_saved_to_disk(self):
        #rg = ReportGenerator(savepath='')
        pass

class Test_Partial_Classification_Report(TestCase):
    pass