import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn_evaluation.plot import rank_one_dimensional

X, y = make_classification(n_samples=5000,
                           n_features=6,
                           n_classes=6,
                           n_informative=4)
rank_one_dimensional(X,
                     features=[
                         'Feature1', 'Feature2', 'Feature3', 'Feature4',
                         'Feature5', 'Feature6'
                     ],
                     orientation='h')
plt.show()
