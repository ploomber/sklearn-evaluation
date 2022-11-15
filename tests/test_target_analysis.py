import pytest

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from sklearn_evaluation import plot


def test_invalid_target():
    """
    A value error should be raised on invalid train or test target
    """
    y_valid = np.random.randint(2, size=100)
    y_invalid = np.random.uniform(size=100)
    with pytest.raises(TypeError):
        plot.target_plot(y_invalid)
    with pytest.raises(TypeError):
        plot.target_plot(y_valid, y_invalid)


def test_class_names_must_match(target_analysis_binary):
    """
    Assert error raised when more classes are in data than specified
    """

    _, _, y_train, y_test = target_analysis_binary

    with pytest.raises(ValueError):
        plot.target_plot(y_train,
                         y_test=y_test,
                         labels=["a", "b", "c"])


def test_binary_balance(target_analysis_binary):
    """
    Test binary classification in balance mode
    """

    _, _, y_train, y_test = target_analysis_binary
    plot.target_plot(y_train)
    #plt.savefig('binary_balance.png')


def test_binary_compare(target_analysis_binary):
    """
    Test binary classification in compare mode
    """

    _, _, y_train, y_test = target_analysis_binary
    plot.target_plot(y_train, y_test)


def test_multiclass_balance(target_analysis_multiclass):
    """
    Test multiclass classification in compare mode
    """
    _, _, y_train, y_test = target_analysis_multiclass
    plot.target_plot(y_train)
    #plt.savefig('multilcass_balance.png')


@image_comparison(baseline_images=['multiclass_compare'],
                  extensions=['png'],
                  remove_text=True)
def test_multiclass_compare(target_analysis_multiclass):
    """
    Test multiclass classification in compare mode
    """
    _, _, y_train, y_test = target_analysis_multiclass
    plot.target_plot(y_train, y_test)
    plt.savefig('multiclass_compare.png', bbox_inches='tight')

