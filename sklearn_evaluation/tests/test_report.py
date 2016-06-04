from unittest import TestCase

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

from sklearn_evaluation import ClassifierEvaluator

from sklearn_evaluation.report import parse_tags


class TestTagParsing(TestCase):
    def test_basic_parse(self):
        tags = parse_tags('{a}{b}{c}')
        self.assertEqual(tags, ['a', 'b', 'c'])

    def test_ignores_tags_with_spaces(self):
        tags = parse_tags('{a}{b}{ c }')
        self.assertEqual(tags, ['a', 'b'])


class TestReportGeneration(TestCase):
    def setUp(self):
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                            iris.target,
                                                            test_size=0.30,
                                                            random_state=0)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)
        target_names = ['setosa', 'versicolor', 'virginica']
        feature_names = range(4)
        model_name = 'a model'

        self.results = ClassifierEvaluator(estimator=model, y_true=y_test,
                                           y_pred=y_pred, y_score=y_score,
                                           feature_names=feature_names,
                                           target_names=target_names,
                                           estimator_name=model_name)

        self.empty = ClassifierEvaluator(estimator=None, y_true=None,
                                         y_pred=None, y_score=None,
                                         feature_names=None,
                                         target_names=None,
                                         estimator_name=None)

        self.template = '''
                            # Report
                            {estimator_type}
                            {date}
                            {confusion_matrix}
                            {roc}
                            {precision_recall}

                            {feature_importances_table}

                            {feature_importances}
                        '''

    def test_no_style(self):
        self.results.generate_report(template=self.template)

    def test_no_style_save(self):
        self.results.generate_report(template=self.template,
                                     path='tmp_report.html')

    def test_generate_with_all_attr_none(self):
        self.empty.generate_report(template='# Title')

    def test_apply_custom_css(self):
        with open('baseline_html/empty.html') as f:
            expected = f.read()

        result = self.empty.generate_report(template='# Title\n ## Section',
                                            style='static/simple.css')

        assert expected == result

    def test_returning_and_saving_are_the_same(self):
        self.empty.generate_report(template='# Title\n ## Section',
                                   path='tmp_report.html')

        with open('tmp_report.html') as f:
            saved = f.read()

        returned = self.empty.generate_report(template='# Title\n ## Section')

        assert saved == returned
