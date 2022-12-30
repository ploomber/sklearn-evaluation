"""
Plotting functions
"""

from .classification import (
    confusion_matrix,
    feature_importances,
    precision_at_proportions,
    ConfusionMatrix,
)

from .roc import roc, ROC
from .precision_recall import precision_recall
from .grid_search import grid_search
from .validation_curve import validation_curve
from .learning_curve import learning_curve
from .metrics import metrics_at_thresholds
from .clustering import (
    elbow_curve,
    elbow_curve_from_results,
    silhouette_analysis,
    silhouette_analysis_from_results,
)
from .regression import residuals, prediction_error, cooks_distance
from .target_analysis import target_analysis
from .calibration import calibration_curve, scores_distribution
from .classification_report import classification_report, ClassificationReport
from .ks_statistics import ks_statistic
from .cumulative_gain_lift_curve import cumulative_gain, lift_curve
from .feature_ranking import Rank1D, Rank2D

__all__ = [
    "confusion_matrix",
    "feature_importances",
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
]
