import numpy as np
from numpy.linalg import svd
from numpy.matlib import repmat
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from warnings import warn
from ..telemetry import SKLearnEvaluationLogger

###### Helper Functions ######
### 1) Compute the singular value decomposition of matrix to get optimal distance cut point for your plot ###


@SKLearnEvaluationLogger.log(feature="plot")
def svd_scree(X):

    """
    Perform singular value decomposition on input matrix. First step in obtaining plottable values for
    the scree plot. Takes care of missing values via mean imputation, returns input array if no missing
    values are detected.

    Parameters
    ----------

    X : array-like, shape = [n_samples, n_features]:
        Data to cluster, where n_samples is the number of samples and
        n_features is the number of features.

    Returns
    -------

    eigvalues : array-like, shape = [n_features, ]
        Array of eigenvalues obtained from singular value decomposition, used to obtain the optimal cluster
        distance in scree plot.

    """

    ## Check for missingness in array, if there is replace the values of the arrays with the mean
    if np.any(np.isnan(X)) == True:
        warn(
            "Missing values detected in input matrix. Replacing missing values with respective column means."
        )

        # compute column means, replace the values of each with the mean of the column
        means = np.mean(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(means, idx[-1])

    else:
        X = X

    ## Perform singular Value Decomposition to get Eigenvectors
    X = np.asmatrix(X.T) * np.asmatrix(X)
    _, s, _ = svd(X)
    eigvalues = s**2 / np.sum(s**2)

    return eigvalues


### 2) Compute the curvature of the curve in order to get maximum distance between points and projected line ###


def best_curvature(eigvalues):
    """
    Helper function returning point with max curvature. Obtained by finding the point of elbow
    curve with maximum distance between line formed by first and last points on the curve. Steps:
    1) we map a vector between our first and last eigenalues, 2) we calculate the distances between our eigenvalues and
    point vector, 3) select the eigenvalue that maximizes distance between the vector and eigenvalue

    Parameters
    ----------
    eigvalues : array-like, shape = [n_features,]
        Array of eigenvalues obtained from singular value decomposition. Eigenvalues are
        descending-ordered - array values and respective indices are utilized to compute the point with max distance


    Returns
    -------
    opt_clusters : int
        Recommended number of clusters for agglomerative clustering dendrogram. Listed as subtitle in scree plot.

    """

    # set up array with curve values
    ticks = len(eigvalues)
    points = np.vstack((range(ticks), eigvalues)).T
    first = points[0]

    # find vector and normalize it
    vector = points[-1] - points[0]
    norm = vector / (np.sqrt(np.sum(vector**2)))

    # return point with shortest distance between vector between first and last point
    values = points - first
    scalar = np.sum(values * repmat(norm, ticks, 1), axis=1)
    vec_parallel = np.outer(scalar, norm)
    vec_line = values - vec_parallel
    dist = np.sqrt(np.sum(vec_line**2, axis=1))
    opt_clusters = np.argmax(dist)

    return opt_clusters


### 3 Scree Plot for optimal cluster cut distance (justification for dendrogram creation) ###


def scree_plot(X, ax=None):
    """
    Plot used to determine the number of clusters in agglomerative clustering. Utilizes two helper functions:
    svd_scree and best_curvature. Respectively, this gives us the values of our plot and the optimal distance for
    cluster cut-point that is suggested to the user.

    Parameters
    ----------

    X : array-like, shape = [n_samples, n_features]:
        Data to cluster, where n_samples is the number of samples and
        n_features is the number of features.

    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes.

    Returns
    ------

    ax : matplotlib Axes
        Axes containing the plot

    """

    # Helper Functions
    eigvalues = svd_scree(X)
    optimal_n = best_curvature(eigvalues)

    # Dimensional Config w/ Title and Subtitle
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Scree Plot -  Agglomerative Cluster Distance Cutoff")

    # Plot scree Curve
    X_ticks = np.arange(X.shape[1])
    plt.plot(X_ticks, eigvalues, marker="o")
    plt.axvline(optimal_n, ls="--", color="black")

    # Setting up axes
    ax.set_xlabel("Principal Component")
    ax.set_xticks(
        X_ticks, [f"{i+1}" for i in X_ticks]
    )  # Labels with Index of eigenvalue + 1
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Optimal N = {optimal_n + 1}")

    return ax


###### Agglomerative Plot ######
### 1) Dendrogram visualizing the creation of our data's heirarchy ###


def aggolmerative_dendrogram(X, model=None, min_clusters=30, ax=None):
    """
    Agglomeratively cluster observations (bottom-up). Helper function to compute the
    scree_plot to show optimal number of clusters that our agglomerative plot needs.

    Parameters
    ----------

    X : array-like, shape = [n_samples, n_features]:
        Data to cluster, where n_samples is the number of samples and
        n_features is the number of features.

    model : Agglomerative clustering instance implementing ``fit`` method,
        ``children_`` and ``distances_`` attributes.
        e.g. :class:`sklearn.linear_model.LinearRegression`. If not specified,
        use AgglomerativeClustering() with 'distance_threshold' equaling optimal n
        eigenvalues from scree plot.

    min_clusters : int
        Minimum number of clusters displayed in dendrogram. In the event that data is very large,
        one can specify values of the dendrogram to clean up its output.

    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes.


    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    # instnantiate model
    model = model or AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=best_curvature(svd_scree(X)),
        linkage="ward",
    )

    # fit data on the classifier passed in, if not specified
    agglomerative = model.fit(X)

    children, distances = agglomerative.children_, agglomerative.distances_
    obs = np.arange(2, children.shape[0] + 2)

    linkage = np.column_stack([children, distances, obs])

    # formatting
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Dendrogram - Agglomerative Clustering")

    # dendrogram
    dendrogram(linkage, truncate_mode="lastp", p=min_clusters)

    ax.legend()
    ax.set_ylabel("Distance")
    ax.set_xlabel("Condensed Dendrogram (Singleton Clusters not in Parentheses)")

    return ax
