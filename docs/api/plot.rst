``sklearn_evaluation.plot``
===========================

The Plot API supports both functional and object-oriented (OOP) interfaces. While the functional 
API allows you to quickly generate out-of-the-box plots and is the easiest to get started with, 
the OOP API offers more flexibility to compare models using a simple synatx, i.e, plot1 + plot2; 
or to customize the style and elements in the plot.


Object Oriented API
*******************


ConfusionMatrix
---------------
.. autoclass:: sklearn_evaluation.plot.ConfusionMatrix
    :members:

PrecisionRecall
----------------
.. autoclass:: sklearn_evaluation.plot.PrecisionRecall
    :members:

ROC
---
.. autoclass:: sklearn_evaluation.plot.ROC
    :members:

ClassificationReport
--------------------
.. autoclass:: sklearn_evaluation.plot.ClassificationReport
    :members:

CalibrationCurve
----------------
.. autoclass:: sklearn_evaluation.plot.CalibrationCurve
    :members:

Rank1D
------
.. autoclass:: sklearn_evaluation.plot.Rank1D
    :members: feature_ranks, feature_ranks_custom_algorithm

Rank2D
------
.. autoclass:: sklearn_evaluation.plot.Rank2D
    :members: feature_ranks, feature_ranks_custom_algorithm



Functional API
**************


calibration_curve
-----------------
.. autofunction:: sklearn_evaluation.plot.calibration_curve

classification_report
---------------------
.. autofunction:: sklearn_evaluation.plot.classification_report

confusion_matrix
----------------
.. autofunction:: sklearn_evaluation.plot.confusion_matrix

cumulative_gain
---------------
.. autofunction:: sklearn_evaluation.plot.cumulative_gain

elbow_curve
-----------
.. autofunction:: sklearn_evaluation.plot.elbow_curve

.. _elbow-curve-from-results-label:

elbow_curve_from_results
------------------------
.. autofunction:: sklearn_evaluation.plot.elbow_curve_from_results

feature_importances
-------------------
.. autofunction:: sklearn_evaluation.plot.feature_importances

grid_search
-----------
.. autofunction:: sklearn_evaluation.plot.grid_search

ks_statistic
------------
.. autofunction:: sklearn_evaluation.plot.ks_statistic

learning_curve
--------------
.. autofunction:: sklearn_evaluation.plot.learning_curve

lift_curve
----------
.. autofunction:: sklearn_evaluation.plot.lift_curve

metrics_at_thresholds
---------------------
.. autofunction:: sklearn_evaluation.plot.metrics_at_thresholds

pca
---
.. autofunction:: sklearn_evaluation.plot.pca

precision_at_proportions
------------------------
.. autofunction:: sklearn_evaluation.plot.precision_at_proportions

precision_recall
----------------
.. autofunction:: sklearn_evaluation.plot.precision_recall


prediction_error
----------------
.. autofunction:: sklearn_evaluation.plot.prediction_error

residuals
---------
.. autofunction:: sklearn_evaluation.plot.residuals

roc
---
.. autofunction:: sklearn_evaluation.plot.roc

scores_distribution
-------------------
.. autofunction:: sklearn_evaluation.plot.scores_distribution

silhouette_analysis
-------------------
.. autofunction:: sklearn_evaluation.plot.silhouette_analysis

.. _silhouette-analysis-from-results-label:

silhouette_analysis_from_results
--------------------------------
.. autofunction:: sklearn_evaluation.plot.silhouette_analysis_from_results

target_analysis
---------------
.. autofunction:: sklearn_evaluation.plot.target_analysis

validation_curve
----------------
.. autofunction:: sklearn_evaluation.plot.validation_curve

cooks_distance
--------------
.. autofunction:: sklearn_evaluation.plot.cooks_distance

report_evaluation
-----------------
.. autofunction:: sklearn_evaluation.report.evaluate_model

report_comparison
-----------------
.. autofunction:: sklearn_evaluation.report.compare_models
