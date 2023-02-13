from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn_evaluation import plot

# generate data
X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
)

kmeans = KMeans(random_state=1, n_init=5)

# plot silhouette analysis of provided clusters
plot.silhouette_analysis(X, kmeans, range_n_clusters=[3])
