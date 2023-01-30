"""
Plotting functions
"""

from sklearn_evaluation.plot.classification import (
    confusion_matrix,
    feature_importances,
    precision_at_proportions,
    ConfusionMatrix,
)

from sklearn_evaluation.plot.roc import roc, ROC
from sklearn_evaluation.plot.precision_recall import precision_recall, PrecisionRecall
from sklearn_evaluation.plot.grid_search import grid_search
from sklearn_evaluation.plot.validation_curve import validation_curve
from sklearn_evaluation.plot.learning_curve import learning_curve
from sklearn_evaluation.plot.metrics import metrics_at_thresholds
from sklearn_evaluation.plot.clustering import (
    elbow_curve,
    elbow_curve_from_results,
    silhouette_analysis,
    silhouette_analysis_from_results,
)

from sklearn_evaluation.plot.regression import (
    residuals,
    prediction_error,
    cooks_distance,
)
from sklearn_evaluation.plot.target_analysis import target_analysis
from sklearn_evaluation.plot.calibration import calibration_curve, scores_distribution
from sklearn_evaluation.plot.classification_report import (
    classification_report,
    ClassificationReport,
)
from sklearn_evaluation.plot.ks_statistics import ks_statistic
from sklearn_evaluation.plot.cumulative_gain_lift_curve import (
    cumulative_gain,
    lift_curve,
)
from sklearn_evaluation.plot.feature_ranking import Rank1D, Rank2D
from sklearn_evaluation.plot.pca import pca

__all__ = [
    "confusion_matrix",
    "feature_importances",
    "PrecisionRecall",
    "precision_recall",
    "roc",
    "ROC",
    "precision_at_proportions",
    "grid_search",
    "validation_curve",
    "learning_curve",
    "metrics_at_thresholds",
    "ConfusionMatrix",
    "residuals",
    "prediction_error",
    "elbow_curve",
    "elbow_curve_from_results",
    "silhouette_analysis",
    "silhouette_analysis_from_results",
    "calibration_curve",
    "target_analysis",
    "scores_distribution",
    "classification_report",
    "ClassificationReport",
    "ks_statistic",
    "cumulative_gain",
    "lift_curve",
    "Rank1D",
    "Rank2D",
    "cooks_distance",
    "pca",
]
