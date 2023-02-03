import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn_evaluation import plot

# generate data
X, y = make_regression(
    n_samples=100,
    n_features=14,
    n_informative=6,
    bias=1.2,
    noise=49.8,
    tail_strength=0.6,
    random_state=637,
)

# plot cooks distance
plot.cooks_distance(X, y)
plt.show()
