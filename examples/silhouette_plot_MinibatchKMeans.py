from sklearn.cluster import MiniBatchKMeans

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

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

kmeans = MiniBatchKMeans(
    n_clusters=2, random_state=0, batch_size=6, max_iter=10, n_init=5
).fit(X)

# plot silhouette analysis
plot.silhouette_analysis(X, kmeans, range_n_clusters=[3, 4])
