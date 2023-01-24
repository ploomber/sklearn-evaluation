from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_evaluation.telemetry import SKLearnEvaluationLogger
from ploomber_core.exceptions import modify_exceptions

import numpy as np
import matplotlib.pyplot as plt


def _set_ax_settings(ax, ind1, ind2, targets=None):
    ax.set_title("Principal Component Plot", fontsize=10)
    ax.set_xlabel(f"Principal Component {ind1}", fontsize=10)
    ax.set_ylabel(f"Principal Component {ind2}", fontsize=10)
    if targets is not None:
        ax.legend(targets)
    ax.grid()


def _validate_inputs(X, n_components, target_names, colors, ax):
    if np.isnan(X).sum():
        raise ValueError("X array should not consist nan " "pca")

    if n_components < 2:
        raise ValueError("n_components should be >= 2")

    if target_names and colors and len(target_names) != len(colors):
        raise ValueError(
            f"Length received. target_names : {len(target_names)}, "
            f"colors : {len(colors)}. Lengths should be same"
        )

    if ax is not None:
        ax_count = int((n_components * (n_components - 1)) / 2)
        if len(ax) != ax_count:
            raise ValueError(f"Number of axes passed should be {ax_count}")


@modify_exceptions
@SKLearnEvaluationLogger.log(feature="plot")
def pca(X, y=None, target_names=None, n_components=2, colors=None, ax=None):
    """
    Plot principle component analysis curve.

    Parameters
    ----------

    X : array-like, shape = [n_samples, n_features]
        Training data, where n_samples is the number of samples and
        n_features is the number of features

    y : array-like or list or None
        set None if ignored otherwise, pass the targets here

    target_names: list, optional
        list of target variable names

    n_components : int, float or 'mle', default=2
        Number of components to keep. If 2, it generates
        the plot of first component vs second component.
        If >=3, all pairwise comparisons are generated.

    colors: list, optional
        colors to be used for the scatter plots for each target.
        If not passed random colors are generated.

    ax : list of matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses current Axes.
        If passed the list should have (n_components * n_components-1)/2
        Axes objects

    Notes
    -----
    .. versionadded:: 0.10.1

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../examples/pca.py

    """

    _validate_inputs(X, n_components, target_names, colors, ax)

    # Standardizing the features
    X = StandardScaler().fit_transform(X)

    if ax is None:
        ax = _generate_axes(n_components)

    # PCA Projection to 2D
    _pca(X, y, target_names, n_components, colors, ax)

    return ax


def _generate_axes(n_components):
    ax = []
    ax_count = int((n_components * (n_components - 1)) / 2)
    for i in range(ax_count):
        _, axes = plt.subplots()
        ax.append(axes)
    return ax


def _plot_generic(n_components, principal_components, ax):
    ax_ind = 0
    for pc1 in range(n_components):
        for pc2 in range(pc1 + 1, n_components):
            ax[ax_ind].scatter(
                principal_components[:, pc1], principal_components[:, pc2], s=50
            )
            _set_ax_settings(ax[ax_ind], pc1 + 1, pc2 + 1, targets=None)
            ax_ind += 1


def _plot_with_target(n_components, target_indices, principal_components, targets, ax):
    ax_ind = 0
    for pc1 in range(n_components):
        for pc2 in range(pc1 + 1, n_components):
            for target in target_indices:
                indices_to_keep, color = target
                ax[ax_ind].scatter(
                    principal_components[indices_to_keep, pc1],
                    principal_components[indices_to_keep, pc2],
                    c=color,
                    s=50,
                )
            _set_ax_settings(ax[ax_ind], pc1 + 1, pc2 + 1, targets)
            ax_ind += 1


def _pca(X, y=None, target_names=None, n_components=2, colors=None, ax=None):

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)

    if y is None:
        _plot_generic(n_components, principal_components, ax)
    else:
        y = y if isinstance(y, np.ndarray) else np.array(y)

        if len(y.shape) < 2:
            y = np.expand_dims(y, 1)
        new_principal_components = np.hstack((principal_components, y))

        from random import randint

        targets = np.unique(new_principal_components[..., -1])
        colors = (
            ["#%06X" % randint(0, 0xFFFFFF) for i in range(len(targets))]
            if colors is None
            else colors
        )

        target_indices = []

        for target, color in zip(targets, colors):
            indices_to_keep = new_principal_components[..., -1] == target
            target_indices.append((indices_to_keep, color))

        targets = target_names if target_names else targets
        _plot_with_target(
            n_components, target_indices, new_principal_components, targets, ax
        )

    return ax
