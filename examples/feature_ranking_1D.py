import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn_evaluation.plot import rank_one_dimensional

X, y = make_classification(n_samples=500,
                           n_features=5,
                           n_classes=3,
                           n_informative=3)
rank_one_dimensional(
    X,
    features=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'],
    orientation='v')
plt.show()
