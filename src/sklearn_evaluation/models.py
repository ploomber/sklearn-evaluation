from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
import re
from sklearn_evaluation.evaluator import _gen_ax
from sklearn_evaluation import plot
from sklearn_evaluation.report.util import (
    check_array_balance,
    get_model_prediction_time,
    get_roc_auc,
    Range,
    validate_args_not_none,
)
from sklearn_evaluation.report import ModelHeuristics, ReportSection

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
    to the step name.

    For example
    ``mh.evaluation_state['auc']``

    The steps are : 'balance', 'accuracy', 'auc', and 'general_stats'.
    """

    def __init__(self, model):
        self.model = model
        super().__init__()

    @validate_args_not_none
    def evaluate_balance(self, y_true):
        """
        Checks if model is balanced

        If minority class is < 5%, then a plot and class imbalance warning is displayed
        """
        balance_section = ReportSection("balance")

        is_balanced = check_array_balance(y_true)

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

    @validate_args_not_none
    def evaluate_accuracy(self, y_true, y_pred_test):
        """
        Measures how many labels the model got right out of the
        total number of predictions

        If accuracy score is < 80%, then a plot and class accuracy warning is displayed
        If accuracy score is > 80% and class is imbalanced, then a plot and class
        accuracy warning is displayed
        """
        accuracy_section = ReportSection("accuracy")

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
                    "evaluate_accuracy", self.model.__class__.__name__, exc=exc
                )
            )

        self._add_section_to_report(accuracy_section)

    @validate_args_not_none
    def evaluate_auc(self, y_true, y_score):
        """
        Checks if roc auc is in acceptable range

        If auc is < 70%, then plot and class low auc warning is displayed
        """
        auc_section = ReportSection("auc")

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

    @validate_args_not_none
    def generate_general_stats(self, y_true, y_pred, y_score):
        """
        Include general stats in the report

        Plots confusion matrix and roc curve
        """
        general_section = ReportSection("general_stats")

        general_section.append_guideline(
            plot.confusion_matrix(y_true, y_pred, ax=_gen_ax())
        )
        general_section.append_guideline(plot.roc(y_true, y_score, ax=_gen_ax()))
        self._add_section_to_report(general_section)


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
    ``mh.evaluation_state['auc']``

    The steps are : 'percision_recall', 'auc', 'prediction_time',
    'calibration', and 'combined_confusion_matrix'

    If model calculation failed an error is displayed
    """

    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        super().__init__()

    @validate_args_not_none
    def precision_and_recall(self, X_test, y_true):
        """
        Calculates precision and recall for each of the models
        """
        percision_recall_section = ReportSection("percision_recall")
        pr_a = None
        pr_b = None

        try:
            y_prob_a = self.model_a.predict_proba(X_test)
            pr_a = plot.PrecisionRecall.from_raw_data(
                y_true, y_prob_a, label=self._get_model_name(self.model_a)
            )

            percision_recall_section.append_guideline(pr_a.ax_)

        except AttributeError as exc:
            percision_recall_section.append_guideline(
                self._get_calculate_failed_error(
                    "percision_recall", self._get_model_name(self.model_a), exc=exc
                )
            )

        try:
            y_prob_b = self.model_b.predict_proba(X_test)
            pr_b = plot.PrecisionRecall.from_raw_data(
                y_true, y_prob_b, label=self._get_model_name(self.model_b)
            )
            percision_recall_section.append_guideline(pr_b.ax_)

        except AttributeError as exc:
            percision_recall_section.append_guideline(
                self._get_calculate_failed_error(
                    "percision_recall", self._get_model_name(self.model_b), exc=exc
                )
            )

        if pr_a and pr_b:
            pr_combined = pr_a + pr_b
            percision_recall_section.append_guideline("Combined PR")
            percision_recall_section.append_guideline(pr_combined.ax_)

        self._add_section_to_report(percision_recall_section)

    @validate_args_not_none
    def auc(self, X_test, y_true):
        """
        Compares models roc auc and adds a report section
        """
        auc_section = ReportSection("auc")

        try:
            y_score_a = self.model_a.predict_proba(X_test)
            roc_auc_model_a = get_roc_auc(y_true, y_score_a)

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
            roc_auc_model_b = get_roc_auc(y_true, y_score_b)

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

    @validate_args_not_none
    def computation(self, X_test):
        """
        Compares models prediction compute time in seconds and
        adds a report section
        """
        computation_section = ReportSection("prediction_time")

        model_a_compute_time = get_model_prediction_time(self.model_a, X_test)
        model_b_compute_time = get_model_prediction_time(self.model_b, X_test)

        compute_time_diff_threshold = 1  # 1 second
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

    @validate_args_not_none
    def calibration(self, X_test, y_true):
        """
        Compares models calibration and adds a report section
        """
        calibration_section = ReportSection("calibration")

        try:
            y_prob_a = self.model_a.predict_proba(X_test)
            p = plot.CalibrationCurve.from_raw_data(
                [y_true],
                [y_prob_a],
                label=[self._get_model_name(self.model_a)],
            ).ax_
            calibration_section.append_guideline(p)
        except AttributeError as exc:
            calibration_section.append_guideline(
                self._get_calculate_failed_error(
                    "calibration", self._get_model_name(self.model_a), exc=exc
                )
            )

        try:
            y_prob_b = self.model_b.predict_proba(X_test)
            p = plot.CalibrationCurve.from_raw_data(
                [y_true],
                [y_prob_b],
                label=[self._get_model_name(self.model_b)],
            ).ax_
            calibration_section.append_guideline(p)
        except AttributeError as exc:
            calibration_section.append_guideline(
                self._get_calculate_failed_error(
                    "calibration", self._get_model_name(self.model_b), exc=exc
                )
            )

        self._add_section_to_report(calibration_section)

    @validate_args_not_none
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

    @validate_args_not_none
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
                    "percision_recall",
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

    report = me.create_report(f"Model evaluation - {model.__class__.__name__}")
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
