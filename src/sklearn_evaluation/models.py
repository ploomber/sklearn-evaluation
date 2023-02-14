from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, auc
import numpy as np
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

COMMUNITY = ("If you need help understanding these stats, " +
             "send us a message on <a href='{COMMUNITY_LINK}'" +
             "target='_blank'>slack</a>")


class ModelEvaluator(ModelHeuristics):
    """
    Generates model evaluation report
    """

    def __init__(self, model):
        self.model = model
        super().__init__()

    @validate_args_not_none
    def evaluate_balance(self, y_true):
        """
        Checks if model is balanced
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
        Measures how many labels the
        model got right out of the total number of predictions
        """
        accuracy_section = ReportSection("accuracy")

        try:
            accuracy_threshold = 0.9
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
    def evaluate_confusion_matrix(y_true, y_pred):
        """
        Checks how many of a classifier's predictions were correct,
        and when incorrect, where the classifier got confused
        """
        # TODO: Implement
        cm = confusion_matrix(y_true, y_pred)

        # normalize
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # TODO: Evaluate model by confusion matrix
        diagonal = np.diagonal(cm)  # noqa

    @validate_args_not_none
    def evaluate_auc(self, y_true, y_score):
        """
        Checks if roc auc is in acceptable range
        """

        if y_true is None or y_score is None:
            return

        auc_section = ReportSection("auc")

        auc_threshold_low_range = Range(0, 0.6)
        auc_threshold_acceptable_range = Range(0.7, 0.8)

        # auc - roc
        roc = plot.ROC.from_raw_data(y_true, y_score)
        for i in range(len(roc.fpr)):
            roc_auc = auc(roc.fpr[i], roc.tpr[i])

            # todo: better check
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
        """
        general_section = ReportSection("general_stats")

        general_section.append_guideline(
            plot.confusion_matrix(y_true, y_pred, ax=_gen_ax())
        )
        general_section.append_guideline(plot.roc(y_true, y_score, ax=_gen_ax()))
        self._add_section_to_report(general_section)


class ModelComparer(ModelHeuristics):
    """
    Compares models and generate report
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
                y_true, y_prob_a, label=self._get_model_name(self.model_a))

            percision_recall_section.append_guideline(pr_a.ax_)

        except Exception as exc:
            percision_recall_section.append_guideline(
                self._get_calculate_failed_error(
                    "percision_recall", self._get_model_name(self.model_a), exc=exc)
            )

        try:
            y_prob_b = self.model_b.predict_proba(X_test)
            pr_b = plot.PrecisionRecall.from_raw_data(
                y_true, y_prob_b, label=self._get_model_name(self.model_b))
            percision_recall_section.append_guideline(pr_b.ax_)

        except Exception as exc:
            percision_recall_section.append_guideline(
                self._get_calculate_failed_error(
                    "percision_recall", self._get_model_name(self.model_b), exc=exc)
            )

        if pr_a and pr_b:
            pr_combined = pr_a + pr_b
            percision_recall_section.append_guideline("Combined PR")
            percision_recall_section.append_guideline(pr_combined.ax_)

        self._add_section_to_report(percision_recall_section)

    @validate_args_not_none
    def auc(self, X_test, y_true):
        """
        Compares models roc auc
        """
        auc_section = ReportSection("auc")

        try:
            y_score_a = self.model_a.predict_proba(X_test)
            roc_auc_model_a = get_roc_auc(y_true, y_score_a)

            if len(roc_auc_model_a) > 1:
                auc_section.append_guideline(
                    f"{self._get_model_name(self.model_a)} AUC (ROC) are {roc_auc_model_a}")
            else:
                auc_section.append_guideline(
                    f"{self._get_model_name(self.model_a)} AUC (ROC) is {roc_auc_model_a[0]}"
                )
        except Exception as exc:
            auc_section.append_guideline(
                self._get_calculate_failed_error(
                    "auc", {self._get_model_name(self.model_a)}, exc=exc)
            )

        try:
            y_score_b = self.model_b.predict_proba(X_test)
            roc_auc_model_b = get_roc_auc(y_true, y_score_b)

            if len(roc_auc_model_b) > 1:
                auc_section.append_guideline(
                    f"{self._get_model_name(self.model_b)} AUC (ROC) are {roc_auc_model_b}")
            else:
                auc_section.append_guideline(
                    f"{self._get_model_name(self.model_b)} AUC (ROC) is {roc_auc_model_b[0]}"
                )
        except Exception as exc:
            auc_section.append_guideline(
                self._get_calculate_failed_error(
                    "auc", {self._get_model_name(self.model_b)}, exc=exc)
            )

        self._add_section_to_report(auc_section)

    @validate_args_not_none
    def computation(self, X_test):
        """
        Compares models prediction compute time
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
                    f"{self._get_model_name(self.model_a)} is a lot more computational expensive"
                )
            else:
                computation_section.append_guideline(
                    f"{self._get_model_name(self.model_b)} is a lot more computational expensive"
                )

        computation_section.append_guideline(
            f"{self._get_model_name(self.model_a)} compute time is {model_a_compute_time} (seconds)"
        )
        computation_section.append_guideline(
            f"{self._get_model_name(self.model_b)} compute time is {model_b_compute_time} (seconds)"
        )

        self._add_section_to_report(computation_section)

    @validate_args_not_none
    def calibration(self, X_test, y_true):
        """
        Compares models calibration
        """
        calibration_section = ReportSection("calibration")

        try:
            y_prob_a = self.model_a.predict_proba(X_test)
            p = plot.CalibrationCurve.from_raw_data([y_true], [y_prob_a],
                                                    label=[self._get_model_name(
                                                        self.model_a)],
                                                    ).ax_
            calibration_section.append_guideline(p)
        except Exception as exc:
            calibration_section.append_guideline(
                self._get_calculate_failed_error(
                    "calibration", self._get_model_name(self.model_a), exc=exc)
            )

        try:
            y_prob_b = self.model_b.predict_proba(X_test)
            p = plot.CalibrationCurve.from_raw_data([y_true], [y_prob_b],
                                                    label=[self._get_model_name(
                                                        self.model_b)],
                                                    ).ax_
            calibration_section.append_guideline(p)
        except Exception as exc:
            calibration_section.append_guideline(
                self._get_calculate_failed_error(
                    "calibration", self._get_model_name(self.model_b), exc=exc)
            )

        self._add_section_to_report(calibration_section)

    @validate_args_not_none
    def add_combined_cm(self, X_test, y_true):
        combined_confusion_matrix_section = ReportSection("combined_confusion_matrix")

        y_score_a = self.model_a.predict(X_test)
        y_score_b = self.model_b.predict(X_test)

        model_a_cm = plot.ConfusionMatrix.from_raw_data(y_true, y_score_a)
        model_b_cm = plot.ConfusionMatrix.from_raw_data(y_true, y_score_b)

        combined = model_a_cm + model_b_cm
        combined_confusion_matrix_section.append_guideline(combined.plot())
        self._add_section_to_report(combined_confusion_matrix_section)

    @validate_args_not_none
    def add_combined_pr(self, X_test, y_true):
        combined_pr_section = ReportSection("combined_pr")

        try:
            y_prob_a = self.model_a.predict_proba(X_test)
            pr_a = plot.PrecisionRecall.from_raw_data(y_true, y_prob_a)

            y_prob_b = self.model_b.predict_proba(X_test)
            pr_b = plot.PrecisionRecall.from_raw_data(y_true, y_prob_b)
            pr_combined = pr_a + pr_b
            combined_pr_section.append_guideline(pr_combined.ax_)

            self._add_section_to_report(combined_pr_section)

        except Exception as exc:
            combined_pr_section.append_guideline(
                self._get_calculate_failed_error(
                    "percision_recall", f"{self._get_model_name(self.model_a)} or {self._get_model_name(self.model_b)}", exc=exc
                )
            )


def evaluate_model(model, y_true, y_pred, y_score=None):
    _check_model(model)
    _check_inputs(y_true, y_pred)
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


def compare_models(model_a, model_b, X_train, X_test, y_true):
    _check_model(model_a)
    _check_model(model_b)

    mc = ModelComparer(model_a, model_b)

    mc.precision_and_recall(X_test, y_true)

    mc.auc(X_test, y_true)

    mc.computation(X_test)

    mc.calibration(X_test, y_true)

    mc.add_combined_cm(X_test, y_true)

    # mc.add_combined_pr(X_test, y_true)

    report = mc.create_report(
        f"Compare models - {mc._get_model_name(model_a)} vs {mc._get_model_name(model_b)}")
    return report


def _check_model(model) -> None:
    """
    Validate if model supported

    Raises
    ~~~~~~
    ModelNotSupported or ValueError?
    """
    if model is None:
        raise ValueError("Model is none")


def _check_inputs(y_true, y_pred) -> None:
    """
    Validate if inputs supported

    Raises
    ~~~~~~
    ModelNotSupported or ValueError?
    """
    # TODO: Implement
    # TODO: If optional args given test them
    pass
