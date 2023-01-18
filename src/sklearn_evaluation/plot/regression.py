"""
Plotting functions for regression plots.

NOTE: Cook's distance is based on the yellowbricks regressor module. License below.

Copyright 2016-2020 The scikit-yb developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn_evaluation.telemetry import SKLearnEvaluationLogger
from sklearn.linear_model import LinearRegression
from ploomber_core.exceptions import modify_exceptions


def _set_ax_settings(ax, xlabel, ylabel, title):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _check_parameter_validity(y_true, y_pred):
    if any((val is None for val in (y_true, y_pred))):
        raise ValueError("y_true and y_pred are needed to plot " "Residuals Plot")

    if y_true.shape != y_pred.shape:
        raise ValueError("parameters should have same shape.")


@SKLearnEvaluationLogger.log(feature="plot")
@modify_exceptions
def residuals(y_true, y_pred, ax=None):
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
    .. plot:: ../examples/residuals.py

    """

    _check_parameter_validity(y_true, y_pred)

    if ax is None:
        _, ax = plt.subplots()

    # horizontal line for residual=0
    ax.axhline(y=0)

    ax.scatter(y_pred, y_true - y_pred)

    _set_ax_settings(ax, "Predicted Value", "Residuals", "Residuals Plot")
    return ax


@SKLearnEvaluationLogger.log(feature="plot")
@modify_exceptions
def prediction_error(y_true, y_pred, ax=None):
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
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../examples/prediction_error.py

    """
    _check_parameter_validity(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots()

    model = LinearRegression()

    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    y_reshaped = y_true.reshape((-1, 1))

    # it is necessary to fit the model with y_true and y_pred
    # to get the best fit line representing the error trend
    model.fit(y_reshaped, y_pred)
    x = np.linspace(min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred)))
    y = model.intercept_ + model.coef_ * x
    ax.plot(x, y, "-b", label="best fit")

    # identity line
    ax.plot(x, x, "--k", label="identity")

    # scatter plot
    ax.scatter(y_true, y_pred)

    # R2
    r2 = model.score(y_reshaped, y_pred)
    plt.plot([], [], " ", label=f"R2 = {round(r2,5)}")

    _set_ax_settings(ax, "y_true", "y_pred", "Prediction Error")
    ax.legend(loc="upper left")
    return ax


@SKLearnEvaluationLogger.log(feature="plot")
@modify_exceptions
def cooks_distance(X, y, ax=None):
    """Plots cooks distance.

    Parameters
    ----------
    X : array-like, 2D
        Training data
        Refer https://numpy.org/doc/stable/glossary.html#term-array-like

    y : array-like, 1D
        Target data
        Refer https://numpy.org/doc/stable/glossary.html#term-array-like

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../examples/cooks_distance.py

    Notes
    -----
    .. versionadded:: 0.8.4
    """
    model = LinearRegression()
    model.fit(X, y)
    leverage = (X * np.linalg.pinv(X).T).sum(1)
    rank = np.linalg.matrix_rank(X)
    df = X.shape[0] - rank
    residuals = y - model.predict(X)
    mse = np.dot(residuals, residuals) / df
    residuals_studentized = residuals / np.sqrt(mse) / np.sqrt(1 - leverage)
    distance_ = residuals_studentized**2 / X.shape[1]
    distance_ *= leverage / (1 - leverage)
    influence_threshold_ = 4 / X.shape[0]
    outlier_percentage_ = sum(distance_ > influence_threshold_) / X.shape[0]
    outlier_percentage_ *= 100.0

    if ax is None:
        _, ax = plt.subplots()
    _, _, baseline = ax.stem(
        distance_, linefmt="C0-", markerfmt=",", use_line_collection=True
    )

    ax.set_xlim(0, len(distance_))

    label = r"{:0.2f}% > $I_t$ ($I_t=\frac {{4}} {{n}}$)".format(outlier_percentage_)
    ax.axhline(
        influence_threshold_,
        ls="--",
        label=label,
        c=baseline.get_color(),
        lw=baseline.get_linewidth(),
    )
    ax.set_title("Cook's Distance Outlier Detection")
    ax.set_xlabel("instance index")
    ax.set_ylabel("influence (I)")
    ax.legend(loc="best", frameon=True)
    return ax
