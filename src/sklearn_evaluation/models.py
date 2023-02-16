from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
import re
from sklearn_evaluation.evaluator import _gen_ax
from sklearn_evaluation import plot
from sklearn_evaluation.report.util import (
    Range,
    run_if_args_are_not_none,
)
from sklearn_evaluation.report import ModelHeuristics, ReportSection
import time
import numpy as np

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
            plot.confusion_matrix(y_true, y_pred, ax=_gen_ax())
        )
        general_section.append_guideline(plot.roc(y_true, y_score, ax=_gen_ax()))
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


class ModelComparer(ModelHeuristics):
    """
    Model comparison helper

    This is a utility class that simplifies the comparison of multiple models
    in one place. You can utilize this comparator to compare different models, e.g
    RandomForestClassifier against DecisionTreeClassifier, or LogisticRegression
    agains LinearRegression.

    The results of each step can be accessed using the `key` that corresponds
    to the step name.

    For example
    ``mh.evaluation_state['precision_recall']``

    The steps are : 'precision_recall', 'auc', 'prediction_time',
    'calibration', and 'combined_confusion_matrix'

    If model calculation failed an error is displayed
    """

    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b

        self.evaluator_a = ModelEvaluator(model_a)
        self.evaluator_b = ModelEvaluator(model_b)

        super().__init__()

    @run_if_args_are_not_none
    def precision_and_recall(self, X_test, y_true):
        """
        Calculates precision and recall for each of the models
        """
        precision_recall_section = ReportSection("precision_recall")

        pr_a = self.evaluator_a.evaluate_precision_and_recall(
            X_test, y_true, precision_recall_section
        )

        pr_b = self.evaluator_b.evaluate_precision_and_recall(
            X_test, y_true, precision_recall_section
        )

        if pr_a and pr_b:
            pr_combined = pr_a + pr_b
            precision_recall_section.append_guideline("Combined PR")
            precision_recall_section.append_guideline(pr_combined.ax_)

        self._add_section_to_report(precision_recall_section)

    @run_if_args_are_not_none
    def auc(self, X_test, y_true):
        """
        Compares models roc auc and adds a report section
        """
        auc_section = ReportSection("auc")

        try:
            y_score_a = self.model_a.predict_proba(X_test)
            roc_auc_model_a = self.evaluator_a.get_roc_auc(y_true, y_score_a)

            if len(roc_auc_model_a) > 1:
                auc_section.append_guideline(
                    f"{self._get_model_name(self.model_a)} "
                    + f"AUC (ROC) : {roc_auc_model_a}"
                )
            else:
                auc_section.append_guideline(
                    f"{self._get_model_name(self.model_a)} "
                    + f"AUC (ROC) is {roc_auc_model_a[0]}"
                )
        except AttributeError as exc:
            auc_section.append_guideline(
                self._get_calculate_failed_error(
                    "auc", self._get_model_name(self.model_a), exc=exc
                )
            )

        try:
            y_score_b = self.model_b.predict_proba(X_test)
            roc_auc_model_b = self.evaluator_b.get_roc_auc(y_true, y_score_b)

            if len(roc_auc_model_b) > 1:
                auc_section.append_guideline(
                    f"{self._get_model_name(self.model_b)} "
                    + f"AUC (ROC) : {roc_auc_model_b}"
                )
            else:
                auc_section.append_guideline(
                    f"{self._get_model_name(self.model_b)} "
                    + f"AUC (ROC) is {roc_auc_model_b[0]}"
                )
        except AttributeError as exc:
            auc_section.append_guideline(
                self._get_calculate_failed_error(
                    "auc", self._get_model_name(self.model_b), exc=exc
                )
            )

        self._add_section_to_report(auc_section)

    @run_if_args_are_not_none
    def computation(self, X_test):
        """
        Compares models prediction compute time in seconds and
        adds a report section

        If time differences > 60 sec, warning is displayed
        """
        computation_section = ReportSection("prediction_time")

        model_a_compute_time = self.evaluator_a.get_model_prediction_time(X_test)
        model_b_compute_time = self.evaluator_b.get_model_prediction_time(X_test)

        compute_time_diff_threshold = 60  # 60 seconds
        is_significant_time_diff = (
            abs(model_a_compute_time - model_b_compute_time)
            >= compute_time_diff_threshold
        )
        if is_significant_time_diff:
            if model_a_compute_time > model_b_compute_time:
                computation_section.append_guideline(
                    f"{self._get_model_name(self.model_a)} "
                    + "is a lot more computational expensive"
                )
            else:
                computation_section.append_guideline(
                    f"{self._get_model_name(self.model_b)} "
                    + "is a lot more computational expensive"
                )

        computation_section.append_guideline(
            f"{self._get_model_name(self.model_a)} "
            + f"compute time is {model_a_compute_time} (seconds)"
        )
        computation_section.append_guideline(
            f"{self._get_model_name(self.model_b)} "
            + f"compute time is {model_b_compute_time} (seconds)"
        )

        self._add_section_to_report(computation_section)

    @run_if_args_are_not_none
    def calibration(self, X_test, y_true):
        """
        Compares models calibration and adds a report section
        """
        calibration_section = ReportSection("calibration")

        self.evaluator_a.evaluate_calibration(X_test, y_true, calibration_section)
        self.evaluator_b.evaluate_calibration(X_test, y_true, calibration_section)

        self._add_section_to_report(calibration_section)

    @run_if_args_are_not_none
    def add_combined_cm(self, X_test, y_true):
        """
        Adds a report guideline with a confusion matrix of model a and model b
        """
        combined_confusion_matrix_section = ReportSection("combined_confusion_matrix")

        try:
            y_score_a = self.model_a.predict(X_test)
            y_score_b = self.model_b.predict(X_test)

            model_a_cm = plot.ConfusionMatrix.from_raw_data(y_true, y_score_a)
            model_b_cm = plot.ConfusionMatrix.from_raw_data(y_true, y_score_b)

            combined = model_a_cm + model_b_cm
            combined_confusion_matrix_section.append_guideline(combined.plot())
        except ValueError as exc:
            combined_confusion_matrix_section.append_guideline(
                self._get_calculate_failed_error(
                    "combined_confusion_matrix",
                    f"{self._get_model_name(self.model_a)} "
                    + f"or {self._get_model_name(self.model_b)}",
                    exc=exc,
                )
            )

        self._add_section_to_report(combined_confusion_matrix_section)

    @run_if_args_are_not_none
    def add_combined_pr(self, X_test, y_true):
        """
        Adds a report guideline with a precision and recall of model a and model b
        """
        combined_pr_section = ReportSection("combined_pr")

        try:
            y_prob_a = self.model_a.predict_proba(X_test)
            pr_a = plot.PrecisionRecall.from_raw_data(y_true, y_prob_a)

            y_prob_b = self.model_b.predict_proba(X_test)
            pr_b = plot.PrecisionRecall.from_raw_data(y_true, y_prob_b)
            pr_combined = pr_a + pr_b
            combined_pr_section.append_guideline(pr_combined.ax_)

        except Exception as exc:
            combined_pr_section.append_guideline(
                self._get_calculate_failed_error(
                    "precision_recall",
                    f"{self._get_model_name(self.model_a)} "
                    + f"or {self._get_model_name(self.model_b)}",
                    exc=exc,
                )
            )

        self._add_section_to_report(combined_pr_section)


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
    _check_model(model)
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


def compare_models(model_a, model_b, X_test, y_true):
    """
    Compares two models and generates an HTML report

    Parameters
    -----------
    model_a : estimator
        An estimator to compare.

    model_b : estimator
        An estimator to compare.

    X_test : array-like of shape (n_samples, n_features)
        Training data, where `n_samples` is the number of samples
        and `n_features` is the number of features.

    y_true : array-like
        Correct target values (ground truth).

    Examples
    --------
    .. seealso:: :ref:`Report: Comparison`

    Compare DecisionTreeClassifier and RandomForestClassifier

    .. plot:: ../examples/report_comparison.py

    Notes
    -----
    .. versionadded:: 0.11.2
    """
    _check_model(model_a)
    _check_model(model_b)

    mc = ModelComparer(model_a, model_b)

    mc.precision_and_recall(X_test, y_true)

    mc.auc(X_test, y_true)

    mc.computation(X_test)

    mc.calibration(X_test, y_true)

    mc.add_combined_cm(X_test, y_true)

    mc.add_combined_pr(X_test, y_true)

    report = mc.create_report(
        f"Compare models - {mc._get_model_name(model_a)} "
        + f"vs {mc._get_model_name(model_b)}"
    )
    return report


def _check_model(model) -> None:
    """
    Validate model

    Raises
    ~~~~~~
    ValueError is model is None

    # TODO: Should we add ModuleNotSupportedError?
    """
    if model is None:
        raise ValueError("Model is none")
