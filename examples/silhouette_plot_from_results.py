from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn_evaluation import plot

X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
)

cluster_labels = []

# Cluster labels for four clusters
kmeans = KMeans(n_clusters=4, n_init=5)
cluster_labels.append(kmeans.fit_predict(X))

# Cluster labels for five clusters
kmeans = KMeans(n_clusters=5, n_init=5)
cluster_labels.append(kmeans.fit_predict(X))


plot.silhouette_analysis_from_results(X, cluster_labels)
