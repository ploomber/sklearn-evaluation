import matplotlib.pyplot as plt

from sklearn_evaluation.plot.util import requires_properties
from sklearn_evaluation.report.serialize import EvaluatorHTMLSerializer
from sklearn_evaluation.report.report import Report
from .util import estimator_type, class_name
from . import plot
from .telemetry import SKLearnEvaluationLogger


class ClassifierEvaluator(object):
    """
    Encapsulates results from an estimator on a testing set to provide a
    simplified API from other modules. All parameters are optional, just
    fill the ones you need for your analysis.

    Parameters
    ----------
    estimator : sklearn estimator
        Must have a ``feature_importances_`` attribute.
    y_true : array-like
        Target predicted classes (estimator predictions).
    y_pred : array-like
        Correct target values (ground truth).
    y_score : array-like
        Target scores (estimator predictions).
    feature_names : array-like
        Feature names.
    target_names : list
        List containing the names of the target classes
    estimator_name : str
        Identifier for the model. This can be later used to identify the
        estimator when generating reports.
    """
    TEMPLATE_NAME = 'classifier.md'

    def __init__(self, estimator=None, y_true=None, y_pred=None, y_score=None,
                 feature_names=None, target_names=None, estimator_name=None,
                 X=None):
        self._estimator = estimator
        self._y_true = y_true
        self._y_pred = y_pred
        self._y_score = y_score
        self._feature_names = feature_names
        self._target_names = target_names
        self._estimator_name = estimator_name
        self._X = X
        # TODO: perform basic logic checking,
        # raise Exception if necessary

    @property
    def estimator_type(self):
        """Estimator name (e.g. RandomForestClassifier)
        """
        return estimator_type(self.estimator)

    @property
    def estimator_class(self):
        """Estimator class (e.g. sklearn.ensemble.RandomForestClassifier)
        """
        return class_name(self.estimator)

    @property
    def estimator(self):
        return self._estimator

    @property
    def X(self):
        return self._X

    @property
    def y_true(self):
        return self._y_true

    @property
    def y_pred(self):
        # get predictions if possible
        if (self._y_pred is None and self.estimator is not None
                and self.X is not None):
            self._y_pred = self.estimator.predict(self.X)
        return self._y_pred

    @property
    def y_score(self):
        # get scores if possible
        if (self._y_score is None and self.estimator is not None
                and self.X is not None):
            self._y_score = self.estimator.predict_proba(self.X)
        return self._y_score

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def target_names(self):
        return self._target_names

    @property
    def estimator_name(self):
        return self._estimator_name

    @requires_properties(('y_true', 'y_pred'))
    def confusion_matrix(self):
        """Confusion matrix plot
        """
        return plot.confusion_matrix(self.y_true, self.y_pred,
                                     self.target_names, ax=_gen_ax()
                                     )

    @requires_properties(('y_true', 'y_score'))
    def roc(self):
        """ROC plot
        """
        return plot.roc(self.y_true, self.y_score, ax=_gen_ax()
                        )

    @requires_properties(('y_true', 'y_score'))
    def precision_recall(self):
        """Precision-recall plot
        """
        return plot.precision_recall(self.y_true, self.y_score, ax=_gen_ax()
                                     )

    @requires_properties(('estimator',))
    def feature_importances(self):
        """Feature importances plot
        """
        return plot.feature_importances(self.estimator,
                                        feature_names=self.feature_names,
                                        ax=_gen_ax()
                                        )

    @requires_properties(('estimator',))
    def feature_importances_table(self):
        """Feature importances table
        """
        from . import table

        return table.feature_importances(self.estimator,
                                         feature_names=self.feature_names)

    @requires_properties(('y_true', 'y_score'))
    def precision_at_proportions(self):
        """Precision at proportions plot
        """
        return plot.precision_at_proportions(self.y_true, self.y_score,
                                             ax=_gen_ax()
                                             )

    def html_serializable(self):
        """
        Returns a EvaluatorHTMLSerializer instance, which is an object with the
        same methods and properties than a ClassifierEvaluator, but it returns
        HTML serialized versions of each
        (i.e. evaluator.feature_importances_table() returns a string with the
        table in HTML format, evaluator.confusion_matrix() returns a HTML image
        element with the image content encoded in base64), useful for
        generating reports using some template system
        """
        return EvaluatorHTMLSerializer(self)

    @SKLearnEvaluationLogger.log(feature='report')
    def make_report(self, template=None):
        """
        Make HTML report

        Parameters
        ----------
        template: str, or pathlib.Path, optional
            HTML or Markdown template with jinja2 format. If a pathlib.Path
            object is passed, the content of the file is read. Within the
            template, the evaluator is passed as "e", so you can use things
            like {{e.confusion_matrix()}} or any other attribute/method. If
            None, a default template is used

        style: str
            Path to a css file to apply style to the report. If None, no
            style will be applied

        Returns
        -------
        Report
            Returns the contents of the report if path is None.

        """
        return Report(self.html_serializable(), template)


def _gen_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return ax
