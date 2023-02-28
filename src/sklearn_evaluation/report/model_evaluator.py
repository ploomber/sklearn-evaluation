from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
import re
from sklearn_evaluation import plot
from sklearn_evaluation.report.util import (
    Range,
    run_if_args_are_not_none,
    gen_ax,
    check_model
)
from sklearn_evaluation.report import ModelHeuristics, ReportSection
import time
import numpy as np
from sklearn_evaluation.telemetry import SKLearnEvaluationLogger


COMMUNITY_LINK = "https://ploomber.io/community"

COMMUNITY = (
    "If you need help understanding these stats, "
    + f"send us a message on <a href='{COMMUNITY_LINK}'"
    + "target='_blank'>slack</a>"
)


class ModelEvaluator(ModelHeuristics):
    """
    Model evaluation report

    This is a utility class that simplifies the evaluation of various model
    aspects, including balance, AUC, and accuracy. Use this evaluator
    to assess different models like RandomForestClassifier, DecisionTreeClassifier,
    LogisticRegression, and LinearRegression.

    The results of each step can be accessed using the `key` that corresponds
    to the step name. Use a custom ReportSection to use your own keys and
    a different report structure.

    For example
    ``mh.evaluation_state['auc']``

    The steps are : 'balance', 'accuracy', 'auc', 'general_stats',
    'calibration', and 'precision_recall'
    """

    def __init__(self, model):
        self.model = model
        super().__init__()

    @run_if_args_are_not_none
    def evaluate_balance(self, y_true, custom_section=None):
        """
        Checks if model is balanced

        If minority class is < 5%, then a plot and class imbalance warning is displayed
        """
        balance_section = custom_section or ReportSection("balance")

        is_balanced = self.check_array_balance(y_true)

        if is_balanced:
            balance_section.set_is_ok(True)
            balance_section.append_guideline("your model is balanced")
        else:
            balance_section.set_is_ok(False)
            p = plot.target_analysis(y_true)
            balance_section.append_guideline("Your test set is highly imbalanced")
            balance_section.append_guideline(p)
            balance_section.append_guideline(COMMUNITY)

        self._add_section_to_report(balance_section)

    @run_if_args_are_not_none
    def evaluate_accuracy(self, y_true, y_pred_test, custom_section=None):
        """
        Measures how many labels the model got right out of the
        total number of predictions

        If accuracy score is < 80%, then a plot and class accuracy warning are displayed
        If accuracy score is > 80% and class is imbalanced, then a plot and class
        accuracy warning are displayed
        """
        accuracy_section = custom_section or ReportSection("accuracy")

        try:
            accuracy_threshold = 0.8
            accuracy = accuracy_score(y_true, y_pred_test)

            accuracy_section.append_guideline(f"Accuracy is {accuracy}")
            if accuracy >= accuracy_threshold:
                if "balance" in self.evaluation_state:
                    balance = self.evaluation_state["balance"]
                    if balance["is_ok"]:
                        accuracy_section.append_guideline["is_ok"] = True
                        accuracy_section.append_guideline("You model is accurate")
                    else:
                        accuracy_section.set_is_ok(False)
                        accuracy_section.append_guideline(
                            "Please note your model is unbalanced, "
                            "so high accuracy could be misleading"
                        )
        except Exception as exc:
            accuracy_section.append_guideline(
                self._get_calculate_failed_error(
                    "evaluate_accuracy", self._get_model_name(self.model), exc=exc
                )
            )

        self._add_section_to_report(accuracy_section)

    @run_if_args_are_not_none
    def evaluate_auc(self, y_true, y_score, custom_section=None):
        """
        Checks if roc auc is in acceptable range

        If auc is < 70%, then plot and low auc warning are displayed
        """
        auc_section = custom_section or ReportSection("auc")

        auc_threshold_low_range = Range(0, 0.7)
        auc_threshold_acceptable_range = Range(0.7, 0.8)

        # auc - roc
        roc = plot.ROC.from_raw_data(y_true, y_score)
        for i in range(len(roc.fpr)):
            roc_auc = auc(roc.fpr[i], roc.tpr[i])

            # TODO: better check
            label = roc.label[i] if len(roc.label) > 0 else f"class {i}"
            r = re.match(r"^\(class (.)*\)", label)
            if r:
                class_name = r[0].replace("(", "").replace(")", "")
            else:
                class_name = label

            if auc_threshold_low_range.in_range(roc_auc):
                auc_section.append_guideline(
                    f"Area under curve is low for {class_name}"
                )
                class_roc = plot.ROC(roc.fpr[i], roc.tpr[i], label=[label]).plot().ax
                auc_section.append_guideline(class_roc)
                auc_section.append_guideline(COMMUNITY)
            elif auc_threshold_acceptable_range.in_range(roc_auc):
                auc_section.set_is_ok(True)
                auc_section.set_include_in_report(False)
            else:
                auc_section.set_is_ok(True)
                auc_section.set_include_in_report(False)

        self._add_section_to_report(auc_section)

    @run_if_args_are_not_none
    def generate_general_stats(self, y_true, y_pred, y_score, custom_section=None):
        """
        Add confusion matrix and roc curve to the report
        """
        general_section = custom_section or ReportSection("general_stats")

        general_section.append_guideline(
            plot.confusion_matrix(y_true, y_pred, ax=gen_ax())
        )
        general_section.append_guideline(plot.roc(y_true, y_score, ax=gen_ax()))
        self._add_section_to_report(general_section)

    @run_if_args_are_not_none
    def evaluate_precision_and_recall(self, X_test, y_true, custom_section=None):
        """
        Returns a precision and recall plot
        """
        precision_recall_section = custom_section or ReportSection("precision_recall")
        try:
            y_prob = self.model.predict_proba(X_test)
            pr = plot.PrecisionRecall.from_raw_data(
                y_true, y_prob, label=self._get_model_name(self.model)
            )

            precision_recall_section.append_guideline(pr.ax_)

        except AttributeError as exc:
            pr = None
            precision_recall_section.append_guideline(
                self._get_calculate_failed_error(
                    "precision_recall", self._get_model_name(self.model), exc=exc
                )
            )

        return pr

    @run_if_args_are_not_none
    def evaluate_calibration(self, X_test, y_true, custom_section=None):
        """
        Returns a calibration plot
        """
        calibration_section = custom_section or ReportSection("calibration")
        try:
            y_prob = self.model.predict_proba(X_test)
            calibration_plot = plot.CalibrationCurve.from_raw_data(
                [y_true],
                [y_prob],
                label=[self._get_model_name(self.model)],
            )
            calibration_section.append_guideline(calibration_plot.ax_)
        except AttributeError as exc:
            calibration_plot = None
            calibration_section.append_guideline(
                self._get_calculate_failed_error(
                    "calibration", self._get_model_name(self.model), exc=exc
                )
            )

        return calibration_plot

    @run_if_args_are_not_none
    def get_roc_auc(self, y_test, y_score) -> list:
        """
        Returns list of roc auc
        """
        roc_auc = []
        roc = plot.ROC.from_raw_data(y_test, y_score)
        for i in range(len(roc.fpr)):
            roc_auc.append(auc(roc.fpr[i], roc.tpr[i]))

        return roc_auc

    @run_if_args_are_not_none
    def get_model_prediction_time(self, X) -> float:
        """
        Returns model.predict(X) time in seconds
        """
        start = time.time()
        self.model.predict(X)
        end = time.time()
        eval_time = end - start  # in seconds
        return eval_time

    @run_if_args_are_not_none
    def check_array_balance(self, array) -> bool:
        """
        Checks if array is balanced.

        Balance threshold is 0.05
        """
        classes, counts = np.unique(array, return_counts=True)
        n_classes = len(classes)
        n_values = len(array)
        expected_balance = 1 / n_classes

        weights = list(map(lambda count: count / n_values, counts))
        balance_threshold = 0.05
        expected_range = Range(
            expected_balance - balance_threshold, expected_balance + balance_threshold
        )

        return all(expected_range.in_range(w) for w in weights)


@SKLearnEvaluationLogger.log(feature="report", action="evaluate_model")
def evaluate_model(model, y_true, y_pred, y_score=None):
    """
    Evaluates a given model and generates an HTML report

    Parameters
    -----------
    model : estimator
        An estimator to evaluate.

    y_true : array-like
        Correct target values (ground truth).

    y_pred : array-like
        Target predicted classes (estimator predictions).

    y_score : array-like, default None
        Target scores (estimator predictions).

    Examples
    --------

    .. seealso:: :ref:`Report: Evaluation`

    Generate evaluation report for RandomForestClassifier

    .. plot:: ../examples/report_evaluation.py

    Notes
    -----
    .. versionadded:: 0.11.2
    """
    check_model(model)
    me = ModelEvaluator(model)

    # check imbalance
    me.evaluate_balance(y_true)

    # accuracy score
    me.evaluate_accuracy(y_true, y_pred)

    # auc
    me.evaluate_auc(y_true, y_score)

    # add general stats
    me.generate_general_stats(y_true, y_pred, y_score)

    report = me.create_report(f"Model evaluation - {me._get_model_name(model)}")
    return report
