import numpy as np
import matplotlib.pyplot as plt


def learning_curve(train_scores, test_scores, train_sizes, ax=None):

    if ax is None:
        ax = plt.gca()

    ax.grid()
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score mean")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")

    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    ax.legend(loc="best")
    ax.margins(0.05)
    return ax
