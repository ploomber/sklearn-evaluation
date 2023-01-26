import time
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn_evaluation import plot

X, _ = make_blobs(n_samples=100, centers=3, n_features=5, random_state=0)

n_clusters = range(1, 30)
sum_of_squares = []
cluster_times = []
for i in n_clusters:
    start = time.time()
    kmeans = KMeans(n_clusters=i, n_init=5)
    sum_of_squares.append(kmeans.fit(X).score(X))
    cluster_times.append(time.time() - start)

sum_of_squares = np.absolute(sum_of_squares)
plot.elbow_curve_from_results(n_clusters, sum_of_squares, cluster_times)
