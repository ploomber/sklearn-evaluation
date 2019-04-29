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


# Testing missing input - checking that meaningful errors are raised when
# trying to plot somethign with missing input (specially relevant when)
# using the report generator


class TestMissingInput(TestCase):

    def test_confusion_matrix(self):
        with self.assertRaisesRegexp(ValueError, "needed to plot"):
            plot.confusion_matrix(None, [1, 0])

    def test_roc(self):
        with self.assertRaisesRegexp(ValueError, "needed to plot"):
            plot.roc([1, 0], None)

    def test_precision_recall(self):
        with self.assertRaisesRegexp(ValueError, "needed to plot"):
            plot.precision_recall(None, None)

    def test_feature_importances(self):
        with self.assertRaisesRegexp(ValueError, "needed to plot"):
            plot.feature_importances(None)

    def test_precision_at_proportions(self):
        with self.assertRaisesRegexp(ValueError, "needed to plot"):
            plot.precision_at_proportions(None, [1, 0])
