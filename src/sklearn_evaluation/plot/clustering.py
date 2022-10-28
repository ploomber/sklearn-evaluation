"""
Plots for clustering models

NOTE: this is largely based in the scikit-plot cluster module. License below.

MIT License

Copyright (c) [2018] [Reiichiro Nakano]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import clone
from joblib import Parallel, delayed

from ..telemetry import SKLearnEvaluationLogger


# TODO: add unit test
@SKLearnEvaluationLogger.log(feature='plot')
def elbow_curve(X,
                clf,
                n_clusters=None,
                n_jobs=1,
                show_cluster_time=True,
                ax=None):
    """Plots elbow curve of different values of K of a clustering algorithm.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]:
        Data to cluster, where n_samples is the number of samples and
        n_features is the number of features.

    clf
        Clusterer instance that implements ``fit``,``fit_predict``, and
        ``score`` methods, and an ``n_clusters`` hyperparameter.
        e.g. :class:`sklearn.cluster.KMeans` instance

    n_clusters : None or :obj:`list` of int, optional
        List of n_clusters for which to plot the explained variances.
        Defaults to ``[1, 3, 5, 7, 9, 11]``.

    n_jobs : int, optional
        Number of jobs to run in parallel. Defaults to 1.

    show_cluster_time : bool, optional
        Include plot of time it took to cluster for a particular K.

    ax : :class:`matplotlib.axes.Axes`, optional
        The axes upon which to plot the curve. If None, the plot is drawn
        on the current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/elbow_curve.py

    """
    if n_clusters is None:
        n_clusters = range(1, 10, 2)
    else:
        n_clusters = sorted(n_clusters)

    if not hasattr(clf, 'n_clusters'):
        raise TypeError('"n_clusters" attribute not in classifier. '
                        'Cannot plot elbow method.')

    tuples = Parallel(n_jobs=n_jobs)(
        delayed(_clone_and_score_clusterer)(clf, X, i) for i in n_clusters)
    clfs, times = zip(*tuples)
    sum_of_squares = np.absolute(clfs)

    return elbow_curve_from_results(n_clusters,
                                    sum_of_squares,
                                    times if show_cluster_time else None,
                                    ax=ax)


def elbow_curve_from_results(n_clusters, sum_of_squares, times, ax=None):
    """
    Same as `elbow_curve`, but it takes the number of clusters and sum of
    squares as inputs. Useful if you want to train the models yourself.
    """
    # TODO: unit test this
    # TODO: also test with unsorted input
    idx = np.argsort(n_clusters)
    n_clusters = np.array(n_clusters)[idx]
    sum_of_squares = np.array(sum_of_squares)[idx]

    if ax is None:
        ax = plt.gca()

    ax.set_title('Elbow Plot')
    ax.plot(n_clusters, sum_of_squares, 'b*-')
    ax.grid(True)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Sum of Squared Errors')

    if times is not None:
        times = np.array(times)[idx]
        ax2_color = 'green'
        ax2 = ax.twinx()
        ax2.plot(n_clusters, times, ':', alpha=0.75, color=ax2_color)
        ax2.set_ylabel('Clustering duration (seconds)',
                       color=ax2_color,
                       alpha=0.75)
        ax2.tick_params(colors=ax2_color)

    return ax


def _clone_and_score_clusterer(clf, X, n_clusters):
    """Clones and scores a clustering model
    """
    start = time.time()
    clf = clone(clf)
    setattr(clf, 'n_clusters', n_clusters)
    return clf.fit(X).score(X), time.time() - start
