from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from sklearn_evaluation import plot

X, _ = make_blobs(n_samples=100, centers=3, n_features=5, random_state=0)
kmeans = KMeans(random_state=1)

plot.silhouette_analysis(X, kmeans)
plt.show()


