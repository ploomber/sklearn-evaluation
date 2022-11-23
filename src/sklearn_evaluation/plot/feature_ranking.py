import numpy as np
import pandas as pd
from scipy.stats import shapiro

import matplotlib.pyplot as plt

ranking_methods = {"shapiro": lambda X: np.array([shapiro(x)[0] for x in X.T])}


def _rank(X, algorithm):
    """
    Returns the feature ranking.
    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features
    algorithm : str or None
        The ranking mechanism to use, or None for the default
    Returns
    -------
    ranks : ndarray
        An n-dimensional, symmetric array of rank scores, where n is the
        number of features. E.g. for 1D ranking, it is (n,), for a
        2D ranking it is (n,n) and so forth.
    """

    algorithm = algorithm.lower()

    if algorithm not in ranking_methods:
        raise ValueError(
            "'{}' is unrecognized ranking method".format(algorithm))

    # Extract matrix from dataframe if necessary
    if isinstance(X, pd.DataFrame):
        X = X.values

    return ranking_methods[algorithm](X)


def _draw(ranks, features, color, orientation, show_feature_names, ax):
    """
    Draws the bar plot of the ranking array of features.
    """
    if orientation == "h":
        # Make the plot
        ax.barh(np.arange(len(ranks)), ranks, color=color)

        # Add ticks and tick labels
        ax.set_yticks(np.arange(len(ranks)))

        if show_feature_names:
            ax.set_yticklabels(features)
        else:
            ax.set_yticklabels([])

        # Order the features from top to bottom on the y axis
        ax.invert_yaxis()

        # Turn off y grid lines
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color='#808080')

    elif orientation == "v":
        # Make the plot
        ax.bar(np.arange(len(ranks)), ranks, color=color)

        # Add ticks and tick labels
        ax.set_xticks(np.arange(len(ranks)))
        if show_feature_names:
            ax.set_xticklabels(features, rotation=90)
        else:
            ax.set_xticklabels([])

        # Turn off x grid lines
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#808080')

    else:
        raise ValueError("Orientation must be 'h' or 'v'")
    return ax


def rank_one_dimensional(X,
                         features=None,
                         algorithm='shapiro',
                         orientation='h',
                         show_feature_names=True,
                         figsize=(7, 7),
                         color='g',
                         ax=None):
    """
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature dataset to be transformed.

    features : list of strings, optional
               list of feature names to be passed if not
               implicit from the input

    algorithm : string, optional
            algorithm to use for ranking the features. Default
                value is shapiro.

    show_feature_names : boolean, optional
            flag to show feature names on chart

    figsize : tuple, optional
            tuple in the format (width, height) to set plot size

    orientation : string 'h' or 'v'
                orientation of the bars. h for horizontal, v for vertical

    color : string, optional
            Specify color of the bar chart

    ax : :class:`matplotlib.axes.Axes`, optional
        The axes upon which to plot the curve. If None, the plot is drawn
        on the current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/feature_rank_1D.py

    """

    n_columns = X.shape[1]

    if features is not None:
        # Use the user-specified features with some checking
        # TODO: allow the user specified features to filter the dataset
        if len(features) != n_columns:
            raise ValueError(
                ("number of supplied feature names does not match the number "
                 "of columns in the training data."))

        features_ = np.array(features)

    else:
        # Attempt to determine the feature names from the input data
        if isinstance(X, pd.DataFrame):
            features_ = np.array(X.columns)

        # Otherwise create numeric labels for each column.
        else:
            features_ = np.arange(0, n_columns)
    ranks_ = _rank(X, algorithm)
    title = "{} Ranking of {} Features".format(algorithm.title(),
                                               len(features_))
    return rank_one_dimensional_from_results(
        ranks_,
        features_,
        orientation=orientation,
        show_feature_names=show_feature_names,
        figsize=figsize,
        color=color,
        title=title,
        ax=ax)


def rank_one_dimensional_from_results(ranks,
                                      features,
                                      orientation='h',
                                      show_feature_names=True,
                                      figsize=(7, 7),
                                      color='g',
                                      title=None,
                                      ax=None):
    if features is None:
        raise ValueError("Please provide feature labels")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax = _draw(ranks, features, color, orientation, show_feature_names, ax)
    if title is None:
        title = "Ranking of {} Features".format(len(features))
    ax.set_title(title)
    return ax
