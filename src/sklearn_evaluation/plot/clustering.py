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
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from sklearn_evaluation.telemetry import SKLearnEvaluationLogger

from ploomber_core.exceptions import modify_exceptions
from warnings import warn

# TODO: add unit test


    
@SKLearnEvaluationLogger.log(feature="plot")
def calinski_harabasz_analysis(
    X,
    clf,
    range_n_clusters=None,
    figsize=None,
    ax=None,
):
    if range_n_clusters is None:
        range_n_clusters = [2, 3, 4, 5, 6]
    else:
        range_n_clusters = sorted(range_n_clusters)

    if not hasattr(clf, "n_clusters"):
        raise TypeError(
            '"n_clusters" attribute not in classifier. '
            "Cannot plot silhouette analysis ."
        )

    for n_clusters in range_n_clusters:
        _, ax = plt.subplots(1, 1, figsize=figsize)
        clf = clone(clf)
        setattr(clf, "n_clusters", n_clusters)
        cluster_labels = clf.fit_predict(X)

        ax = calinski_harabasz_analysis_from_results(
            X, cluster_labels, figsize, ax
        )
    return ax


@SKLearnEvaluationLogger.log(feature="plot")
@modify_exceptions
def calinski_harabasz_analysis_from_results(
    X,
    cluster_labels,
    figsize=None,
    ax=None,
):
    
    cluster_labels = np.asarray(cluster_labels)

    n_clusters = len(np.unique(cluster_labels))

    calinski_harabasz_value = calinski_harabasz_score(X, cluster_labels)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

   
    ax.set_title("calinski_harabasz_score Plot")
    ax.plot(n_clusters, calinski_harabasz_value, "b*-", label="calinski_harabasz_value")
    ax.grid(True)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("calinski_harabasz_value")

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, labels)
    return ax


@SKLearnEvaluationLogger.log(feature="plot")
def davies_bouldin_analysis(
    X,
    clf,
    range_n_clusters=None,
    figsize=None,
    cmap="nipy_spectral",
    text_fontsize="medium",
    ax=None,
):

    if range_n_clusters is None:
        range_n_clusters = [2, 3, 4, 5, 6]
    else:
        range_n_clusters = sorted(range_n_clusters)

    if not hasattr(clf, "n_clusters"):
        raise TypeError(
            '"n_clusters" attribute not in classifier. '
            "Cannot plot silhouette analysis ."
        )

    for n_clusters in range_n_clusters:
        _, ax = plt.subplots(1, 1, figsize=figsize)
        clf = clone(clf)
        setattr(clf, "n_clusters", n_clusters)
        cluster_labels = clf.fit_predict(X)

        ax = davies_bouldin_analysis_from_results(
            X, cluster_labels, figsize, cmap, text_fontsize, ax
        )
    return ax


@SKLearnEvaluationLogger.log(feature="plot")
@modify_exceptions
def davies_bouldin_analysis_from_results(
    X,
    cluster_labels,
    figsize=None,
    ax=None,
):
    
    cluster_labels = np.asarray(cluster_labels)

    n_clusters = len(np.unique(cluster_labels))

    davies_bouldin_value = davies_bouldin_score(X, cluster_labels)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

   

    ax.set_title("davies_bouldin_Index Plot")
    ax.plot(n_clusters, davies_bouldin_value, "b*-", label="davies_bouldin_value")
    ax.grid(True)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("davies_bouldin_value")

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, labels)
    return ax



@SKLearnEvaluationLogger.log(feature="plot")
@modify_exceptions
def elbow_curve(
    X,
    clf,
    range_n_clusters=None,
    n_jobs=1,
    show_cluster_time=True,
    ax=None,
    n_clusters=None
):


    if n_clusters is not None:
        if range_n_clusters is not None:
            raise AttributeError(
                "n_cluster attribute is deprecated. Please use only range_n_clusters."
            )
        else:
            warn(
                "elbow_curve will change its signature."
                " Please use range_n_clusters instead of n_cluster",
                FutureWarning,
                stacklevel=2,
            )

    if range_n_clusters is None:
        range_n_clusters = range(1, 10, 2)
    else:
        range_n_clusters = sorted(range_n_clusters)

    if not hasattr(clf, "n_clusters"):
        raise TypeError(
            '"n_clusters" attribute not in classifier. ' "Cannot plot elbow method."
        )

    tuples = Parallel(n_jobs=n_jobs)(
        delayed(_clone_and_score_clusterer)(clf, X, i) for i in range_n_clusters
    )
    clfs, times = zip(*tuples)
    sum_of_squares = np.absolute(clfs)
    return elbow_curve_from_results(
        range_n_clusters, sum_of_squares, times if show_cluster_time else None, ax=ax
    )

def _clone_and_score_clusterer(clf, X, n_clusters):
    """Clones and scores a clustering model"""
    start = time.time()
    clf = clone(clf)
    setattr(clf, "n_clusters", n_clusters)
    return clf.fit(X).score(X), time.time() - start


@SKLearnEvaluationLogger.log(feature="plot")
@modify_exceptions
def elbow_curve_from_results(n_clusters, sum_of_squares, times, ax=None):

    # TODO: unit test this
    # TODO: also test with unsorted input
    idx = np.argsort(n_clusters)
    n_clusters = np.array(n_clusters)[idx]
    sum_of_squares = np.array(sum_of_squares)[idx]

    if ax is None:
        ax = plt.gca()

    ax.set_title("Elbow Plot")
    ax.plot(n_clusters, sum_of_squares, "b*-", label="Elbow")
    ax.grid(True)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Sum of Squared Errors")

    handles, labels = ax.get_legend_handles_labels()
    if times is not None:
        times = np.array(times)[idx]
        ax2_color = "green"
        ax2 = ax.twinx()
        ax2.plot(
            n_clusters, times, ":", alpha=0.75, color=ax2_color, label="Fitting time"
        )
        ax2.set_ylabel("Clustering duration (seconds)", color=ax2_color, alpha=0.75)
        ax2.tick_params(colors=ax2_color)

        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles + handles2
        labels = labels + labels2

    ax.legend(handles, labels)
    return ax


@SKLearnEvaluationLogger.log(feature="plot")
def silhouette_analysis(
    X,
    clf,
    range_n_clusters=None,
    metric="euclidean",
    figsize=None,
    cmap="nipy_spectral",
    text_fontsize="medium",
    ax=None,
):

    if range_n_clusters is None:
        range_n_clusters = [2, 3, 4, 5, 6]
    else:
        range_n_clusters = sorted(range_n_clusters)

    if not hasattr(clf, "n_clusters"):
        raise TypeError(
            '"n_clusters" attribute not in classifier. '
            "Cannot plot silhouette analysis ."
        )

    for n_clusters in range_n_clusters:
        _, ax = plt.subplots(1, 1, figsize=figsize)
        clf = clone(clf)
        setattr(clf, "n_clusters", n_clusters)
        cluster_labels = clf.fit_predict(X)

        ax = silhouette_analysis_from_results(
            X, cluster_labels, metric, figsize, cmap, text_fontsize, ax
        )
    return ax


@SKLearnEvaluationLogger.log(feature="plot")
@modify_exceptions
def silhouette_analysis_from_results(
    X,
    cluster_labels,
    metric="euclidean",
    figsize=None,
    cmap="nipy_spectral",
    text_fontsize="medium",
    ax=None,
):
    
    cluster_labels = np.asarray(cluster_labels)

    le = LabelEncoder()
    cluster_labels_encoded = le.fit_transform(cluster_labels)

    n_clusters = len(np.unique(cluster_labels))

    silhouette_avg = silhouette_score(X, cluster_labels, metric=metric)

    sample_silhouette_values = silhouette_samples(X, cluster_labels, metric=metric)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_xlim([-0.1, 1])

    if hasattr(X, "shape"):
        length = X.shape[0]
    else:
        length = len(X)
    ax.set_ylim([0, length + (n_clusters + 1) * 10 + 10])

    ax.set_xlabel("Silhouette coefficient values", fontsize=text_fontsize)
    ax.set_ylabel("Cluster label", fontsize=text_fontsize)

    y_lower = 10

    for i in range(n_clusters):
        ax.set_title(f"Silhouette Analysis (n_clusters={n_clusters})", fontsize="large")

        ith_cluster_silhouette_values = sample_silhouette_values[
            cluster_labels_encoded == i
        ]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.get_cmap(cmap)(float(i) / n_clusters)

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax.text(
            -0.05,
            y_lower + 0.5 * size_cluster_i,
            str(le.classes_[i]),
            fontsize=text_fontsize,
        )

        y_lower = y_upper + 10

    ax.axvline(
        x=silhouette_avg,
        color="red",
        linestyle="--",
        label="Silhouette score: {0:0.3f}".format(silhouette_avg),
    )

    ax.set_yticks([])  # Clear the y-axis labels / ticks
    ax.set_xticks(np.arange(-0.1, 1.0, 0.2))

    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc="best", fontsize=text_fontsize)
    return ax






    


    

