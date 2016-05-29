from unittest import TestCase

import numpy as np
from sklearn.preprocessing import label_binarize

from sklearn_evaluation import plot


class TestROC(TestCase):
    def test_1d_array(self):
        # y_true = np.array([1, 1, 1, 0, 0, 0])
        # y_score = np.array([0.7, 0.6, 0.9, 0.6, 0.1, 0.1])
        # plot.roc(y_true, y_score)
        pass

    def test_multiclass(self):
        # y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        # y_score = np.array([0.7, 0.6, 0.9, 0.6, 0.1, 0.1])
        # plot.roc(y_true, y_score)
        pass
