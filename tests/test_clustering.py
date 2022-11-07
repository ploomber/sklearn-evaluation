import numpy as np
import matplotlib.pyplot as plt

from unittest import TestCase
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris as load_data
from sklearn_evaluation import plot


def convert_labels_into_string(y_true):
    return ["A" if x == 0 else x for x in y_true]


class TestPlotSilhouette(TestCase):

    def setUp(self):
        np.random.seed(0)
        self.X, self.y = load_data(return_X_y=True)
        p = np.random.permutation(len(self.X))
        self.X, self.y = self.X[p], self.y[p]

    def tearDown(self):
        plt.close("all")

    def test_plot_silhouette(self):
        np.random.seed(0)
        clf = KMeans()
        plot.silhouette_plot(self.X, clf)

    def test_plot_silhouette_with_cluster_range(self):
        np.random.seed(0)
        clf = KMeans()
        plot.silhouette_plot(self.X, clf, range_n_clusters=[3, 4])

    def test_string_classes(self):
        np.random.seed(0)
        clf = KMeans()
        cluster_labels = clf.fit_predict(self.X)
        plot.silhouette_plot_from_results(
            self.X, convert_labels_into_string(cluster_labels))

    def test_cmap(self):
        np.random.seed(0)
        clf = KMeans()
        cluster_labels = clf.fit_predict(self.X)
        plot.silhouette_plot_from_results(self.X,
                                          cluster_labels,
                                          cmap='Spectral')
        plot.silhouette_plot_from_results(self.X,
                                          cluster_labels,
                                          cmap=plt.cm.Spectral)

    def test_ax(self):
        np.random.seed(0)
        clf = KMeans()
        cluster_labels = clf.fit_predict(self.X)
        plot.silhouette_plot_from_results(self.X, cluster_labels)
        fig, ax = plt.subplots(1, 1)
        out_ax = plot.silhouette_plot_from_results(self.X, cluster_labels)
        assert ax is not out_ax
        out_ax = plot.silhouette_plot_from_results(self.X,
                                                   cluster_labels,
                                                   ax=ax)
        assert ax is out_ax

    def test_array_like(self):
        plot.silhouette_plot_from_results(self.X.tolist(), self.y.tolist())
        plot.silhouette_plot_from_results(self.X.tolist(),
                                          convert_labels_into_string(self.y))
