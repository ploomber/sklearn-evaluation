from sklearn.datasets import load_iris
from matplotlib.testing.decorators import image_comparison

from sklearn_evaluation import plot

X, y = load_iris(return_X_y=True)


@image_comparison(
    baseline_images=["agglomerative_dendrogram"], extensions=["png"], remove_text=True
)
def test_agglomerative_clustering():
    plot.agglomerative_dendrogram(X)


@image_comparison(baseline_images=["scree_plot"], extensions=["png"], remove_text=True)
def test_scree_plot():
    plot.scree_plot(X)
