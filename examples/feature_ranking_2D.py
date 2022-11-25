import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn_evaluation.plot import Rank2D

X, y = make_classification(n_samples=1000,
                           n_features=6,
                           n_classes=2,
                           n_informative=4,
                           class_sep=0.8)

rank2d = Rank2D()
rank2d.plot_feature_ranks(X)
plt.show()
