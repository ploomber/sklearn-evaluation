import numpy as np
import matplotlib.pyplot as plt

from ..telemetry import SKLearnEvaluationLogger
from ..util import is_column_vector

@SKLearnEvaluationLogger.log(feature='plot')
def cooks_distance(X, y_true, y_pred, influence_threshold_constant=4., draw_threshold=True, linefmt='C0-', markerfmt=',', ax=None, **kwargs):
    """Plot Cooks Distance

    Parameters
    ----------
    X : array-like, shape = [n_samples, x_dim]
        Data used to generate the predictions `y_pred`

    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).

    y_pred : array-like, shape = [n_samples]
        Target predicted classes (estimator predictions).

    influence_threshold_constant : float, default: 4.0
        The rule of thumb influence threshold to determine outliers in the regression model,
        defined as It=4.0/n.  
    
    draw_threshold : bool, default: True
        Draw a horizontal line at D(i) == 4/n to easily identify the most influential
        points on the final regression. This will also draw a legend that specifies the
        percentage of data points that are above the threshold.

    linefmt : str, default: 'C0-'
        A string defining the properties of the vertical lines of the stem plot, usually
        this will be a color or a color and a line style. The default is simply a solid
        line with the first color of the color cycle.

    markerfmt : str, default: ','
        A string defining the properties of the markers at the stem plot heads. The
        default is "pixel", e.g. basically no marker head at the top of the stem plot.

    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/cooks_distance.py
    """
    if any((val is None for val in (X, y_true, y_pred))):
         raise ValueError("X, y_true and y_pred are needed to plot Cook's Distance")
    
    if is_column_vector(X):
        x_dim = 1
    else:
        x_dim = X.shape[1]

    if not ax:
        ax = plt.gca()

    n_samples = y_true.shape[0]
    leverage = (X * np.linalg.pinv(X).T).sum(1)

    residuals = y_true - y_pred
    mse = np.dot(residuals, residuals) / (n_samples - np.linalg.matrix_rank(X))
    distances = (residuals ** 2) / (mse * x_dim) * (leverage / (1-leverage)**2)
    
    influence_threshold = influence_threshold_constant / n_samples
   
    outlier_percentage = (sum(distances > influence_threshold) / n_samples) * 100.0

    # plotting
    _, __, baseline = ax.stem(
        distances, 
        linefmt=linefmt, 
        markerfmt=markerfmt, 
        use_line_collection=True,
    )

    ax.set_xlim(0, n_samples)

    if draw_threshold:
        label = r"{:0.2f}% > $I_t$ ($I_t=\frac {{{:0.1f}}} {{n}}$)".format(
                    outlier_percentage, influence_threshold_constant
                )

        ax.axhline(
            influence_threshold,
            ls='--',
            label=label,
            c=baseline.get_color(),
            lw=baseline.get_linewidth(),
        )
        ax.legend(loc="best", frameon=True)


    ax.set_title(kwargs.get("title", "Cook's Distance Outlier Detection"))
    ax.set_xlabel(kwargs.get("xlabel", "Instance index"))
    ax.set_ylabel(kwargs.get("ylabel", "Influence (I)"))

    return ax