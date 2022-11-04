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
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


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


def _clone_and_silhouette_analysis(clf, X, n_clusters):
    """Clones and scores a clustering model using silhouette score
    """
    start = time.time()
    clf = clone(clf)
    setattr(clf, 'n_clusters', n_clusters)
    return clf.fit(X).score(X), time.time() - start


def silhouette_analysis(X,
                        clf,
                        range_n_clusters=None,
                        ):
    if range_n_clusters is None:
        range_n_clusters = [2, 3, 4, 5, 6]
    else:
        range_n_clusters = sorted(range_n_clusters)

    if not hasattr(clf, 'n_clusters'):
        raise TypeError('"n_clusters" attribute not in classifier. '
                        'Cannot plot silhouette analysis .')
    # tuples = Parallel(n_jobs=n_jobs)(
    #     delayed(_clone_and_score_clusterer)(clf, X, i) for i in n_clusters)

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clf = clone(clf)
        setattr(clf, 'n_clusters', n_clusters)
        #clusterer = clf(n_clusters=n_clusters, random_state=10)
        cluster_labels = clf.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        # print(
        #     "For n_clusters =",
        #     n_clusters,
        #     "The average silhouette_score is :",
        #     silhouette_avg,
        # )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clf.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()

