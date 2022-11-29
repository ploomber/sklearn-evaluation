"""
Plotting functions for regression plots.
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn_evaluation.telemetry import SKLearnEvaluationLogger
from sklearn.linear_model import LinearRegression


@SKLearnEvaluationLogger.log(feature='plot')
def residuals(y_true, y_pred, ax=None, plot_name='Residuals Plot'):
    """
    Plot the residuals between measured and predicted values.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Measured target values (ground truth).
    y_pred : array-like, shape = [n_samples]
        Predicted target values.
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/residuals.py

    """

    _check_parameter_validity(y_true, y_pred)

    if ax is None:
        ax = plt.gca()

    # horizontal line for residual=0
    ax.axhline(y=0)

    ax.scatter(y_pred, y_true-y_pred)

    _set_ax_settings(ax, 'Predicted Value', 'Residuals', plot_name)
    return ax

@SKLearnEvaluationLogger.log(feature='plot')
def prediction_error(y_true,
                     y_pred,
                     model=None,
                     ax=None,
                     plot_name='Prediction Error'):
    """
    Plot the scatter plot of measured values v. predicted values, with
    an identity line and a best fitted line to show the prediction
    difference.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Measured target values (ground truth).
    y_pred : array-like, shape = [n_samples]
        Predicted target values.
    model : Regression instance that implements ``fit``,``predict``, and
        ``score`` methods and ``fit_intercept`` attribute.
        e.g. :class:`sklearn.linear_model.LinearRegression` instance
        If not specified, use the LinearRegression model.
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/prediction_error.py

    """

    _check_parameter_validity(y_true, y_pred)
    if ax is None:
        ax = plt.gca()

    model = model or LinearRegression()
    if not hasattr(model, 'fit_intercept'):
        raise TypeError('"fit_intercept" attribute not in model. '
                        'Cannot plot prediction error.')
    # best fit line
    model.fit(y_true.reshape((-1, 1)), y_pred)
    x = np.linspace(80,230,100)
    y = model.intercept_ + model.coef_ * x
    ax.plot(x,y, "-b", label="best fit")

    # identity line
    ax.plot(x,x, "--k", label="identity")

    # scatter plot
    ax.scatter(y_true, y_pred)

    # R2
    r2 = model.score(y_true.reshape((-1, 1)), y_pred)
    plt.plot([], [], ' ', label=f"R2 = {round(r2,5)}")

    _set_ax_settings(ax, 'y_true', 'y_pred', plot_name)
    ax.legend(loc="upper left")
    return ax

def _set_ax_settings(ax, xlabel, ylabel, title):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def _check_parameter_validity(y_true, y_pred):
    if any((val is None for val in (y_true, y_pred))):
        raise ValueError('y_true and y_pred are needed to plot '
                         'Residuals Plot')

    if y_true.shape != y_pred.shape:
        raise ValueError('parameters should have same shape.')
