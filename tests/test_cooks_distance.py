from sklearn.datasets import make_regression
from sklearn_evaluation.plot import cooks_distance

X, y = make_regression(
        n_samples=500,
        n_features=7,
        n_informative=4,
        bias=1.2,
        noise=49.8,
        tail_strength=0.6,
        random_state=10,
    )


def test_cooks_distance():
    

