from . import plots
from .util import gen_ax
from .report import generate


class ClassifierEvaluator(object):
    '''
        This class encapsulates the results of a model that has been trained,
        you can pass and instance of this class to ReportGenerator for
        generating HTML reports. It also performs some basic checks based on
        the properties (e.g. y_true, y_pred and y_score are the same length,
        target_names and values in y_pred have same number of different values)
    '''
    # All parameters are optional, when creating a report, ReportGenerator
    # class should be able to figure out what it can generate based on the
    # parameters
    def __init__(self, model=None, y_true=None, y_pred=None, y_score=None,
                 feature_names=None, target_names=None, model_name=None):
        self._model = model
        self._y_true = y_true
        self._y_pred = y_pred
        self._y_score = y_score
        self._feature_names = feature_names
        self._target_names = target_names
        self._model_name = model_name
        # TODO: perform basic logic checking,
        # raise Exception if necessary

    # Properties should be read-only to ensure instance integrity
    @property
    def model(self):
        return self._model

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
    def model_name(self):
        return self._model_name

    @property
    def confusion_matrix(self):
        return plots.confusion_matrix(self.y_true, self.y_pred,
                                      self.target_names, ax=gen_ax())

    @property
    def roc(self):
        return plots.roc(self.y_true, self.y_score, ax=gen_ax())

    @property
    def precision_recall(self):
        return plots.precision_recall(self.y_true, self.y_score, ax=gen_ax())

    @property
    def feature_importances(self):
        return plots.feature_importances(self.model,
                                         self.model.feature_importances_,
                                         ax=gen_ax())

    @property
    def precision_at_proportions(self):
        return plots.precision_at_proportions(self.y_true, self.y_score,
                                              ax=gen_ax())

    def generate_report(self, template, path):
        return generate(self, template, path)
