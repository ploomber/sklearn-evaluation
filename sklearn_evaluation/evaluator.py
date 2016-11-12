import matplotlib.pyplot as plt

from .util import estimator_type, class_name
from . import plot


class ClassifierEvaluator(object):
    """
    Encapsuates results from an estimator on a testing set to provide a
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
        Target scores (estimador predictions).
    feature_names : array-like
        Feature names.
    target_names : list
        List containing the names of the target classes
    estimator_name : str
        Identifier for the model. This can be later used to idenfity the
        estimator when generaing reports.
    """

    def __init__(self, estimator=None, y_true=None, y_pred=None, y_score=None,
                 feature_names=None, target_names=None, estimator_name=None):
        self._estimator = estimator
        self._y_true = y_true
        self._y_pred = y_pred
        self._y_score = y_score
        self._feature_names = feature_names
        self._target_names = target_names
        self._estimator_name = estimator_name
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
    def y_true(self):
        return self._y_true

    @property
    def y_pred(self):
        return self._y_pred

    @property
    def y_score(self):
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

    @property
    def confusion_matrix(self):
        """Confusion matrix plot
        """
        return plot.confusion_matrix(self.y_true, self.y_pred,
                                     self.target_names, ax=_gen_ax())

    @property
    def roc(self):
        """ROC plot
        """
        return plot.roc(self.y_true, self.y_score, ax=_gen_ax())

    @property
    def precision_recall(self):
        """Precision-recall plot
        """
        return plot.precision_recall(self.y_true, self.y_score, ax=_gen_ax())

    @property
    def feature_importances(self):
        """Feature importances plot
        """
        return plot.feature_importances(self.estimator,
                                        feature_names=self.feature_names,
                                        ax=_gen_ax())

    @property
    def feature_importances_table(self):
        """Feature importances table
        """
        from . import table

        return table.feature_importances(self.estimator,
                                         feature_names=self.feature_names)

    @property
    def precision_at_proportions(self):
        """Precision at proportions plot
        """
        return plot.precision_at_proportions(self.y_true, self.y_score,
                                             ax=_gen_ax())

    def generate_report(self, template, path=None, style=None):
        """
         Generate HTML report

        Parameters
        ----------
        template : markdown-formatted string or path to the template
            file used for rendering the report. Any attribute of this
            object can be included in the report using the {tag} format.
            e.g.'# Report{estimator_name}{roc}{precision_recall}'.
            Apart from every attribute, you can also use {date} and {date_utc}
            tags to include the date for the report generation using local
            and UTC timezones repectively.

        path : str
            Path to save the HTML report. If None, the function will return
            the HTML code.

        style: str
            Path to a css file to apply style to the report. If None, no
            style will be applied

        Returns
        -------
        report: str
            Returns the contents of the report if path is None.

        """
        from .report import generate

        return generate(self, template, path, style)


def _gen_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return ax
