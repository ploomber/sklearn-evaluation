import numpy as np
import matplotlib.pyplot as plt


def validation_curve(train_scores, test_scores, param_range, param_name=None,
                     semilogx=False, ax=None):
    if ax is None:
        ax = plt.gca()

    if semilogx:
        ax.set_xscale('log')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.set_title("Validation Curve")
    ax.set_ylabel("Score mean")

    if param_name:
        ax.set_xlabel(param_name)

    ax.plot(param_range, train_scores_mean, label="Training score", color="r")
    ax.plot(param_range, test_scores_mean, label="Cross-validation score",
            color="g")

    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2, color="r")
    ax.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2, color="g")

    ax.legend(loc="best")
    ax.margins(0.05)
    return ax
