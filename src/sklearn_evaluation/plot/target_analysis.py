import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.multiclass import unique_labels, type_of_target
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def _validate_target(y):
        """
        Raises a value error if the target is not a classification target.
        """
        # Ignore None values
        if y is None:
            return

        y_type = type_of_target(y)
        if y_type not in ("binary", "multiclass"):
            raise TypeError(
                "'{}' target type not supported, only binary and multiclass".format(
                    y_type
                )
            )


def target_plot():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    _validate_target(y_train)
    _validate_target(y_test)
    # Get the unique values from the dataset
    targets = (y_train,) if y_test is None else (y_train, y_test)
    classes_ = unique_labels(*targets)
    print('Classes : {}'.format(classes_))
    support_ = np.array([(y_train == idx).sum() for idx in classes_])
    print('Support : {}'.format(support_))
    ax = plt.gca()
    ax.bar(
        np.arange(len(support_)),
        support_,
        align="center",
        width=0.5,
    )
    ax.set_title("Class Balance")

    # Set the x ticks with the class names or labels if specified
    labels = classes_
    xticks = np.arange(len(labels))

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)

    # Compute the ceil ing for the y limit
    cmax = support_.max()
    ax.set_ylim(0, cmax + cmax * 0.1)
    ax.set_ylabel("support")

    # Remove the vertical grid
    ax.grid(False, axis="x")
    return ax


