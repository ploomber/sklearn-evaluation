"""
Plotting functions for regression plots.
"""

import matplotlib.pyplot as plt

from sklearn_evaluation.telemetry import SKLearnEvaluationLogger


@SKLearnEvaluationLogger.log(feature='plot')
def residual(y_true, y_pred, ax=None):

    _check_parameter_validity(y_true, y_pred)

    if ax is None:
        ax = plt.gca()

    # a line for residual=0
    ax.axhline(y=0)

    ax.scatter(y_pred, y_true-y_pred)

    _set_ax_settings(ax, 'Predicted Value', 'Residuals')
    return ax

@SKLearnEvaluationLogger.log(feature='plot')
def prediction_error(y_true, y_pred, ax=None):
    _check_parameter_validity(y_true, y_pred)
    if ax is None:
        ax = plt.gca()

    ax.scatter(y_pred, y_true)
    _set_ax_settings(ax, 'y_measured', 'y_predicted')
    return ax


def _set_ax_settings(ax, xlabel, ylabel):
    ax.set_title('Residuals Plot')
    # ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.legend(loc="best")

def _check_parameter_validity(y_true, y_pred):
    if any((val is None for val in (y_true, y_pred))):
        raise ValueError('y_true and y_pred are needed to plot '
                         'Residuals Plot')

    # raise error if parameter's dimension != 1?
    # if y_true.ndim != 1 or y_pred.ndim != 1:
    #     raise ValueError('parameters should be one-dimension.')

    if y_true.shape != y_pred.shape:
        raise ValueError('parameters should have same shape.')
