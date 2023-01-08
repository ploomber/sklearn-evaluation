"""
Plots for feature ranking

NOTE: this is based on the yellowbricks feature module. License below.

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

import numpy as np
import pandas as pd
from scipy.stats import shapiro
from scipy.stats import spearmanr
from scipy.stats import kendalltau as sp_kendalltau
from sklearn_evaluation.telemetry import SKLearnEvaluationLogger
from ploomber_core.exceptions import modify_exceptions

import matplotlib.pyplot as plt


def kendalltau(X):
    """
    Accepts a matrix X and returns a correlation matrix so that each column
    is the variable and each row is the observations.
    Parameters
    ----------
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features
    """
    corrs = np.zeros((X.shape[1], X.shape[1]))
    for idx, cola in enumerate(X.T):
        for jdx, colb in enumerate(X.T):
            corrs[idx, jdx] = sp_kendalltau(cola, colb)[0]
    return corrs


class RankD:
    """
    Base visualizer for Rank1D and Rank2D
    Parameters
    ----------
    algorithm : string
        The ranking algorithm to use; options and defaults vary by subclass

    features : list
        A list of feature names to use.
        If a DataFrame is passed as input and features is None, feature
        names are selected as the columns of the DataFrame.

    figsize : tuple, optional
            (width, height) for specifying the size of the plot.

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required)

    Attributes
    ----------
    ranks_ : ndarray
        An n-dimensional, symmetric array of rank scores, where n is the
        number of features. E.g. for 1D ranking, it is (n,), for a
        2D ranking it is (n,n) and so forth.

    Notes
    -----
    .. versionadded:: 0.8.4
    """

    ranking_methods = {}

    def __init__(self, algorithm=None, features=None, figsize=(7, 7), ax=None):

        self.ranks_ = None
        self.algorithm = algorithm
        self.features = features

        if ax is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
        else:
            self.ax = ax

    def _rank(self, X):
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

        algorithm = self.algorithm.lower()

        if algorithm not in self.ranking_methods:
            raise ValueError("'{}' is unrecognized ranking method".format(algorithm))

        # Extract matrix from dataframe if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.ranking_methods[algorithm](X)

    def _derive_features_from_data(self, X):
        n_columns = X.shape[1]

        if self.features is not None:
            if len(self.features) != n_columns:
                raise ValueError(
                    (
                        "number of supplied feature names does not match the number "
                        "of columns in the training data."
                    )
                )

            self.features_ = np.array(self.features)

        else:
            # Attempt to determine the feature names from the input data
            if isinstance(X, pd.DataFrame):
                self.features_ = np.array(X.columns)

            # Otherwise create numeric labels for each column.
            else:
                self.features_ = np.arange(0, n_columns)

    def _derive_features_from_ranks(self, ranks):
        if self.features is None:
            self.features_ = np.arange(0, len(ranks))
        else:
            if len(self.features) != len(ranks):
                raise ValueError(
                    (
                        "number of supplied feature names does not match the number "
                        "of ranks provided."
                    )
                )
            self.features_ = np.array(self.features)

    @SKLearnEvaluationLogger.log(feature="plot")
    @modify_exceptions
    def feature_ranks(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature dataset to be ranked.
            Refer https://numpy.org/doc/stable/glossary.html#term-array-like

        Returns
        ------
        ax: matplotlib Axes
            Axes containing the plot
        """
        self._derive_features_from_data(X)
        self.ranks_ = self._rank(X)
        self._draw()
        return self.ax

    @SKLearnEvaluationLogger.log(feature="plot")
    @modify_exceptions
    def feature_ranks_custom_algorithm(self, ranks):
        """
        This method is useful if user wants to use custom algorithm for feature ranking.

        Parameters
        ----------
        ranks : ndarray
            An n-dimensional, symmetric array of rank scores, where n is the
            number of features. E.g. for 1D ranking, it is (n,), for a
            2D ranking it is (n,n).

        Returns
        ------
        ax: matplotlib Axes
            Axes containing the plot
        """
        self._validate_rank(ranks)
        self._derive_features_from_ranks(ranks)
        self.ranks_ = ranks
        self._draw()
        return self.ax


class Rank1D(RankD):
    """
    Rank1D computes a score for each feature in the data set with a specific
    metric or algorithm (e.g. Shapiro-Wilk) then returns the features ranked
    as a bar plot.

    Parameters
    ----------
    algorithm : one of {'shapiro', }, default: 'shapiro'
        The ranking algorithm to use, default is 'Shapiro-Wilk.

    features : list
        A list of feature names to use.
        If a DataFrame is passed features is None, feature
        names are selected as the columns of the DataFrame.

    figsize : tuple, optional
            (width, height) for specifying the size of the plot.

    orient : 'h' or 'v', default='h'
        Specifies a horizontal or vertical bar chart.

    color: string
        Specify color for barchart

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    Attributes
    ----------
    ranks_ : ndarray
        An array of rank scores with shape (n,), where n is the
        number of features.

    Examples
    ---------
    .. plot:: ../examples/feature_ranking_1D.py

    Notes
    -----
    .. versionadded:: 0.8.4
    """

    ranking_methods = {"shapiro": lambda X: np.array([shapiro(x)[0] for x in X.T])}

    def __init__(
        self,
        algorithm="shapiro",
        features=None,
        figsize=(7, 7),
        orient="h",
        color="g",
        ax=None,
    ):
        super().__init__(algorithm=algorithm, features=features, figsize=figsize, ax=ax)
        self.color = color
        self.orientation_ = orient

    @staticmethod
    def _validate_rank(ranks):
        if ranks.ndim != 1:
            raise ValueError("Ranks must be 1-dimensional")

    def _draw(self):
        """
        Draws the bar plot of the ranking array of features.
        """

        title = "{} Ranking of {} Features".format(
            self.algorithm.title(), len(self.features_)
        )
        self.ax.set_title(title)
        if self.orientation_ == "h":
            # Make the plot
            self.ax.barh(np.arange(len(self.ranks_)), self.ranks_, color=self.color)

            # Add ticks and tick labels
            self.ax.set_yticks(np.arange(len(self.ranks_)))
            self.ax.set_yticklabels(self.features_)
            # Order the features from top to bottom on the y axis
            self.ax.invert_yaxis()
            # Turn off y grid lines
            self.ax.set_axisbelow(True)
            self.ax.xaxis.grid(True, color="#808080")

        elif self.orientation_ == "v":
            # Make the plot
            self.ax.bar(np.arange(len(self.ranks_)), self.ranks_, color=self.color)

            # Add ticks and tick labels
            self.ax.set_xticks(np.arange(len(self.ranks_)))
            self.ax.set_xticklabels(self.features_, rotation=90)

            # Turn off x grid lines
            self.ax.set_axisbelow(True)
            self.ax.yaxis.grid(True, color="#808080")

        else:
            raise ValueError("Orientation must be 'h' or 'v'")
        return self.ax


class Rank2D(RankD):
    """
    Rank2D performs pairwise comparisons of each feature in the data set with
    a specific metric or algorithm (e.g. Pearson correlation) then returns
    them ranked as a lower left triangle diagram.

    Parameters
    ----------

    algorithm : str, default: 'pearson'
        The ranking algorithm to use, one of: 'pearson', 'covariance', 'spearman',
        or 'kendalltau'.

    features : list
        A list of feature names to use.
        If a DataFrame is passed features is None, feature
        names are selected as the columns of the DataFrame.

    colormap : string or cmap, default: 'RdBu_r'
        optional string or matplotlib cmap to colorize lines
        Use either color to colorize the lines on a per class basis or colormap to
        color them on a continuous scale.

    figsize : tuple, optional
            (width, height) for specifying the size of the plot

    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    Attributes
    ----------
    ranks_ : ndarray
        An array of rank scores with shape (n,n), where n is the
        number of features.

    Examples
    ----------
    .. plot:: ../examples/feature_ranking_2D.py

    Notes
    -----
    .. versionadded:: 0.8.4
    """

    ranking_methods = {
        "pearson": lambda X: np.corrcoef(X.transpose()),
        "covariance": lambda X: np.cov(X.transpose()),
        "spearman": lambda X: spearmanr(X, axis=0)[0],
        "kendalltau": lambda X: kendalltau(X),
    }

    def __init__(
        self,
        algorithm="pearson",
        features=None,
        colormap="RdBu_r",
        figsize=(7, 7),
        ax=None,
    ):

        super().__init__(algorithm=algorithm, features=features, figsize=figsize, ax=ax)
        self.colormap = colormap

    @staticmethod
    def _validate_rank(ranks):
        if ranks.ndim != 2:
            raise ValueError("Ranks must be 2-dimensional")

    def _draw(self):
        """
        Draws the heatmap of the ranking matrix of variables.
        """

        title = "{} Ranking of {} Features".format(
            self.algorithm.title(), len(self.features_)
        )
        self.ax.set_title(title)

        # Set the axes aspect to be equal
        self.ax.set_aspect("equal")

        # Generate a mask for the upper triangle
        mask = np.zeros_like(self.ranks_, dtype=bool)
        mask[np.triu_indices_from(mask)] = True

        # Draw the heatmap
        data = np.ma.masked_where(mask, self.ranks_)
        mesh = self.ax.pcolormesh(data, cmap=self.colormap, vmin=-1, vmax=1)

        # Set the Axis limits
        self.ax.set(xlim=(0, data.shape[1]), ylim=(0, data.shape[0]))

        # Add the colorbar
        cb = self.ax.figure.colorbar(mesh, None, self.ax, fraction=0.046, pad=0.04)
        cb.outline.set_linewidth(0)

        # Reverse the rows to get the lower left triangle
        self.ax.invert_yaxis()

        # Add ticks and tick labels
        self.ax.set_xticks(np.arange(len(self.ranks_)) + 0.5)
        self.ax.set_yticks(np.arange(len(self.ranks_)) + 0.5)
        self.ax.set_xticklabels(self.features_, rotation=90)
        self.ax.set_yticklabels(self.features_)
