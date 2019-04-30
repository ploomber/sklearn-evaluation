"""
Tools for binarizing scores
"""
import numpy as np

from sklearn_evaluation import validate


@validate.argument_is_proportion('top_proportion')
def cutoff_score_at_top_proportion(y_score, top_proportion):
    """
    Sort scores and get the score at
    """
    # Sort scores in descending order
    scores_sorted = np.sort(y_score)[::-1]
    # Based on the proportion, get the index to split th
    # if value is negative, return 0
    cutoff_index = max(int(len(y_score) * top_proportion) - 1, 0)
    # Get the cutoff value
    cutoff_value = scores_sorted[cutoff_index]
    return cutoff_value


def cutoff_score_at_top_n(y_score, top_n):
    scores_sorted = np.sort(y_score)[::-1]
    return scores_sorted[top_n]


def cutoff_score_at_quantile(y_score, quantile):
    return np.quantile(y_score, quantile)


@validate.argument_is_proportion('top_proportion')
def scores_at_top_proportion(y_score, top_proportion):
    """Binary scores by sorting them and grabbing a proportion from the top
    """
    cutoff_score = cutoff_score_at_top_proportion(y_score, top_proportion)
    y_score_binary = np.array(y_score >= cutoff_score).astype(int)
    return y_score_binary


def at_top_n(y_score, top_n):
    cutoff_score = cutoff_score_at_top_n(y_score, top_n)
    y_score_binary = np.array(y_score >= cutoff_score).astype(int)
    return y_score_binary


@validate.argument_is_proportion('quantile')
def scores_at_quantile(y_score, quantile):
    """Binary scores at certain quantile
    """
    cutoff_score = cutoff_score_at_quantile(y_score, quantile)
    y_score_binary = (y_score >= cutoff_score).astype(int)
    return y_score_binary


def scores_at_thresholds(y_score, n_thresholds=10, start=0.0):
    """
    Binarize scores at increasing thresholds, outputs a binary
    matrix where each row corresponds to the binary labels
    obtained from thresholding. n thresholds are generated
    uniformly from [start, 1.0]
    """
    thresholds = np.linspace(start, 1.0, n_thresholds)
    Y_score = np.tile(y_score, (n_thresholds, 1))
    Y_pred = (Y_score >= thresholds[:, np.newaxis]).astype(int)
    return thresholds, Y_pred
