from random import shuffle
from unittest import TestCase

import numpy as np

from sklearn_evaluation.preprocessing import binarize


class Test_binarize_scores_at_top_proportion(TestCase):
    def setUp(self):
        self.scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    def test_at_10(self):
        binary_scores = binarize.scores_at_top_proportion(self.scores, 0.1)
        expected = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_equal(binary_scores, expected)

    def test_at_50(self):
        binary_scores = binarize.scores_at_top_proportion(self.scores, 0.5)
        expected = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        np.testing.assert_equal(binary_scores, expected)

    def test_at_100(self):
        binary_scores = binarize.scores_at_top_proportion(self.scores, 1.0)
        expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_equal(binary_scores, expected)

    def test_proportion_less_than_zero(self):
        self.assertRaises(
            ValueError, binarize.scores_at_top_proportion, self.scores, -0.1
        )

    def test_proportion_more_than_one(self):
        self.assertRaises(
            ValueError,
            binarize.scores_at_top_proportion,
            self.scores,
            top_proportion=1.1,
        )


class Test_cutoff_score_at_top_proportion(TestCase):
    def setUp(self):
        self.scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        shuffle(self.scores)

    def test_at_10(self):
        threshold = binarize.cutoff_score_at_top_proportion(self.scores, 0.1)
        self.assertEqual(threshold, 1.0)

    def test_at_50(self):
        threshold = binarize.cutoff_score_at_top_proportion(self.scores, 0.5)
        self.assertEqual(threshold, 0.6)

    def test_at_100(self):
        threshold = binarize.cutoff_score_at_top_proportion(self.scores, 1.0)
        self.assertEqual(threshold, 0.1)

    def test_proportion_less_than_zero(self):
        self.assertRaises(
            ValueError, binarize.cutoff_score_at_top_proportion, self.scores, -0.1
        )

    def test_proportion_more_than_one(self):
        self.assertRaises(
            ValueError,
            binarize.cutoff_score_at_top_proportion,
            self.scores,
            top_proportion=1.1,
        )
