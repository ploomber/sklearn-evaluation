from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from sklearn_evaluation import plot

# generate data
X, _ = make_blobs(n_samples=100, centers=3, n_features=5, random_state=0)
kmeans = KMeans(random_state=1, n_init=5)

# plot elbow curve
plot.elbow_curve(X, kmeans, range_n_clusters=range(1, 30))
plt.show()
