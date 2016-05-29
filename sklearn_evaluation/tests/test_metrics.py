from unittest import TestCase
from sklearn_evaluation.metrics import (precision_at, labels_at,
                                        tp_at, fp_at)

from sklearn_evaluation.metrics import __threshold_at as threshold_at
from sklearn_evaluation.metrics import __binarize_scores_at as binarize_scores_at

import numpy as np
from numpy import nan
from random import shuffle


class Test_threshold_at(TestCase):

    def setUp(self):
        self.scores = np.array(
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        shuffle(self.scores)

    def test_at_10(self):
        threshold = threshold_at(self.scores, 0.1)
        self.assertEqual(threshold, 1.0)

    def test_at_50(self):
        threshold = threshold_at(self.scores, 0.5)
        self.assertEqual(threshold, 0.6)

    def test_at_100(self):
        threshold = threshold_at(self.scores, 1.0)
        self.assertEqual(threshold, 0.1)

    def test_proportion_less_than_zero(self):
        self.assertRaises(ValueError, threshold_at, self.scores, -0.1)

    def test_proportion_more_than_one(self):
        self.assertRaises(
            ValueError, threshold_at, self.scores, proportion=1.1)


class Test_binarize_scores_at(TestCase):

    def setUp(self):
        self.scores = np.array(
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    def test_at_10(self):
        binary_scores = binarize_scores_at(self.scores, 0.1)
        expected = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_equal(binary_scores, expected)

    def test_at_50(self):
        binary_scores = binarize_scores_at(self.scores, 0.5)
        expected = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        np.testing.assert_equal(binary_scores, expected)

    def test_at_100(self):
        binary_scores = binarize_scores_at(self.scores, 1.0)
        expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_equal(binary_scores, expected)

    def test_proportion_less_than_zero(self):
        self.assertRaises(ValueError, binarize_scores_at, self.scores, -0.1)

    def test_proportion_more_than_one(self):
        self.assertRaises(
            ValueError, binarize_scores_at, self.scores, proportion=1.1)


class Test_precision_at(TestCase):

    def test_perfect_precision(self):
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        scores = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        prec, cutoff = precision_at(labels, scores, proportion=0.10)
        self.assertEqual(prec, 1.0)
        self.assertEqual(cutoff, 100)

    def test_perfect_precision_with_nas(self):
        labels = np.array([1, nan, 1, 1, 1, nan, 0, 0, 0, 0])
        scores = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        prec, cutoff = precision_at(
            labels, scores, proportion=0.10, ignore_nas=True)
        self.assertEqual(prec, 1.0)
        self.assertEqual(cutoff, 100)

    def test_baseline_precision(self):
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        scores = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        prec, cutoff = precision_at(labels, scores, proportion=1.0)
        self.assertEqual(prec, 0.5)
        self.assertEqual(cutoff, 10)

    def test_baseline_precision_with_nas(self):
        labels = np.array([nan, 1, nan, 1, 1, nan, nan, 0, 0, 0])
        scores = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        prec, cutoff = precision_at(
            labels, scores, proportion=1.0, ignore_nas=True)
        self.assertEqual(prec, 0.5)
        self.assertEqual(cutoff, 10)

    def test_proportion_less_than_zero(self):
        self.assertRaises(ValueError, precision_at, [1], [0], -0.1)

    def test_proportion_more_than_one(self):
        self.assertRaises(ValueError, precision_at, [1], [0], proportion=1.1)


class Test_labels_at(TestCase):

    def test_no_labels_at_1(self):
        y_true = np.array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
        y_score = np.random.rand(1, 10)
        labels = labels_at(y_true, y_score, proportion=0.01, normalize=False)
        self.assertEqual(labels, 0)

    def test_no_labels_at_50(self):
        y_true = np.array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
        y_score = np.random.rand(1, 10)
        labels = labels_at(y_true, y_score, proportion=0.5, normalize=False)
        self.assertEqual(labels, 0)

    def test_no_labels_at_100(self):
        y_true = np.array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
        y_score = np.random.rand(1, 10)
        labels = labels_at(y_true, y_score, proportion=1.0, normalize=False)
        self.assertEqual(labels, 0)

    def test_one_label_at_10(self):
        y_true = np.array([1, nan, nan, nan, nan, nan, nan, nan, nan, nan])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        labels = labels_at(y_true, y_score, proportion=0.1, normalize=False)
        self.assertEqual(labels, 1)

    def test_one_label_at_10_norm(self):
        y_true = np.array([1, nan, nan, nan, nan, nan, nan, nan, nan, nan])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        labels = labels_at(y_true, y_score, proportion=0.1, normalize=True)
        self.assertEqual(labels, 1.0)

    def test_one_label_at_50(self):
        y_true = np.array([1, nan, nan, nan, nan, nan, nan, nan, nan, nan])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        labels = labels_at(y_true, y_score, proportion=0.5, normalize=False)
        self.assertEqual(labels, 1)

    def test_one_label_at_100(self):
        y_true = np.array([1, nan, nan, nan, nan, nan, nan, nan, nan, nan])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        labels = labels_at(y_true, y_score, proportion=1.0, normalize=False)
        self.assertEqual(labels, 1)

    def test_60_labels_at_60(self):
        y_true = np.array([1, 1, 1, 1, 1, 1, nan, nan, nan, nan])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        labels = labels_at(y_true, y_score, proportion=0.6, normalize=False)
        self.assertEqual(labels, 6)

    def test_60_labels_at_60_norm(self):
        y_true = np.array([1, 1, 1, 1, 1, 1, nan, nan, nan, nan])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        labels = labels_at(y_true, y_score, proportion=0.6, normalize=True)
        self.assertEqual(labels, 1.0)

    def test_60_labels_at_60_mixed_values(self):
        y_true = np.array([1, 0, 0, 1, 0, 1, nan, nan, nan, nan])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        labels = labels_at(y_true, y_score, proportion=0.6, normalize=False)
        self.assertEqual(labels, 6)

    def test_60_labels_at_60_norm_mixed_values(self):
        y_true = np.array([0, 0, 0, 1, 0, 1, nan, nan, nan, nan])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        labels = labels_at(y_true, y_score, proportion=0.6, normalize=True)
        self.assertEqual(labels, 1.0)

    def test_60_labels_at_30(self):
        y_true = np.array([1, 1, 1, 1, 1, 1, nan, nan, nan, nan])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        labels = labels_at(y_true, y_score, proportion=0.3, normalize=False)
        self.assertEqual(labels, 3)

    def test_60_labels_at_30_norm(self):
        y_true = np.array([1, 1, 1, 1, 1, 1, nan, nan, nan, nan])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        labels = labels_at(y_true, y_score, proportion=0.3, normalize=True)
        self.assertEqual(labels, 0.5)

    def test_proportion_less_than_zero(self):
        self.assertRaises(ValueError, labels_at, [1], [0], -0.1)

    def test_proportion_more_than_one(self):
        self.assertRaises(ValueError, labels_at, [1], [0], proportion=1.1)


class Test_tp_at(TestCase):

    def test_with_nas(self):
        y_true = np.array([1, nan, 1, 1, 1, 1, 1, 1, 1, nan])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        tps = tp_at(y_true, y_score, proportion=0.1)
        self.assertEqual(tps, 1)

    def test_all_tp_at_10(self):
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        tps = tp_at(y_true, y_score, proportion=0.1)
        self.assertEqual(tps, 1)

    def test_all_tp_at_50(self):
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        tps = tp_at(y_true, y_score, proportion=0.5)
        self.assertEqual(tps, 5)

    def test_all_tp_at_100(self):
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        tps = tp_at(y_true, y_score, proportion=1.0)
        self.assertEqual(tps, 10)

    def test_no_tp_at_50(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        tps = tp_at(y_true, y_score, proportion=0.5)
        self.assertEqual(tps, 0)

    def test_no_tp_at_100(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        tps = tp_at(y_true, y_score, proportion=1.0)
        self.assertEqual(tps, 0)

    def test_some_tp_at_10(self):
        y_true = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        tps = tp_at(y_true, y_score, proportion=0.1)
        self.assertEqual(tps, 1)

    def test_some_tp_at_50(self):
        y_true = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        tps = tp_at(y_true, y_score, proportion=0.5)
        self.assertEqual(tps, 3)

    def test_some_tp_at_100(self):
        y_true = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        tps = tp_at(y_true, y_score, proportion=1.0)
        self.assertEqual(tps, 4)

    def test_proportion_less_than_zero(self):
        self.assertRaises(ValueError, tp_at, [1], [0], -0.1)

    def test_proportion_more_than_one(self):
        self.assertRaises(ValueError, tp_at, [1], [0], proportion=1.1)


class Test_fp_at(TestCase):

    def test_with_nas(self):
        y_true = np.array([0, nan, 1, 1, 1, 1, 1, 1, 1, nan])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        fps = fp_at(y_true, y_score, proportion=0.1)
        self.assertEqual(fps, 1)

    def test_all_fp_at_10(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        fps = fp_at(y_true, y_score, proportion=0.1)
        self.assertEqual(fps, 1)

    def test_all_fp_at_50(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        fps = fp_at(y_true, y_score, proportion=0.5)
        self.assertEqual(fps, 5)

    def test_all_fp_at_100(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        fps = fp_at(y_true, y_score, proportion=1.0)
        self.assertEqual(fps, 10)

    def test_no_fp_at_50(self):
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        fps = fp_at(y_true, y_score, proportion=0.5)
        self.assertEqual(fps, 0)

    def test_no_fp_at_100(self):
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        fps = fp_at(y_true, y_score, proportion=1.0)
        self.assertEqual(fps, 0)

    def test_some_fp_at_10(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        fps = fp_at(y_true, y_score, proportion=0.1)
        self.assertEqual(fps, 1)

    def test_some_fp_at_50(self):
        y_true = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        fps = fp_at(y_true, y_score, proportion=0.5)
        self.assertEqual(fps, 2)

    def test_some_fp_at_100(self):
        y_true = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])
        y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        fps = fp_at(y_true, y_score, proportion=1.0)
        self.assertEqual(fps, 6)

    def test_proportion_less_than_zero(self):
        self.assertRaises(ValueError, fp_at, [1], [0], -0.1)

    def test_proportion_more_than_one(self):
        self.assertRaises(ValueError, fp_at, [1], [0], proportion=1.1)
