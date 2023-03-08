from sklearn_evaluation import plot
from sklearn_evaluation.report.util import (
    run_if_args_are_not_none,
    check_model
)
from sklearn_evaluation.report import ModelHeuristics, ReportSection, ModelEvaluator
from sklearn_evaluation.telemetry import SKLearnEvaluationLogger


class ModelsComparer(ModelHeuristics):
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
        except Exception as exc:
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
        except Exception as exc:
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


@SKLearnEvaluationLogger.log(feature="report", action="compare_models")
def compare_models(model_a, model_b, X_test, y_true, report_title=None):
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

    report_title : str, default "Compare models - {model_a} vs {model_b}"

    Examples
    --------
    .. seealso:: :ref:`Report: Comparison`

    Compare DecisionTreeClassifier and RandomForestClassifier

    .. plot:: ../examples/report_comparison.py

    Notes
    -----
    .. versionadded:: 0.11.4
    """
    check_model(model_a)
    check_model(model_b)

    mc = ModelsComparer(model_a, model_b)

    mc.precision_and_recall(X_test, y_true)

    mc.auc(X_test, y_true)

    mc.computation(X_test)

    mc.calibration(X_test, y_true)

    mc.add_combined_cm(X_test, y_true)

    mc.add_combined_pr(X_test, y_true)

    report_title = report_title or f"Compare models - {mc._get_model_name(model_a)} " \
        f"vs {mc._get_model_name(model_b)}"

    report = mc.create_report(report_title)
    return report
