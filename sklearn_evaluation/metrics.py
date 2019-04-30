# from collections import OrderedDict

import numpy as np
from sklearn.metrics import precision_score
from . import validate
from . import util


class ThresholdBinarizer:
    """

    Examples
    --------
    >>> from sklearn_evaluation.metrics import ThresholdBinarizer
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.metrics import precision_score, recall_score, f1_score
    >>> import numpy as np
    >>> y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    >>> y_score = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    >>> ThresholdBinarizer.metric_at(accuracy_score, y_true, y_score)
    >>> ThresholdBinarizer.metric_at([precision_score, recall_score,
    >>>                               f1_score], y_true, y_score)
    """
    @staticmethod
    def metric_at(fn, y_true, y_score, n_thresholds=10, start=0.0):
        """
        Given scores, binarize them at different thresholds, then compute
        metrics
        """
        if util.isiter(fn):
            (thresholds,
             Y_pred) = ThresholdBinarizer.at(y_score,
                                             n_thresholds=n_thresholds)
            metrics = [np.array([fn_(y_true, y_pred) for y_pred in Y_pred])
                       for fn_ in fn]
            return thresholds, metrics
        else:
            (thresholds,
             Y_pred) = ThresholdBinarizer.at(y_score,
                                             n_thresholds=n_thresholds)
            metrics = np.array([fn(y_true, y_pred) for y_pred in Y_pred])
            return thresholds, metrics

    @staticmethod
    def at(y_score, n_thresholds=10, start=0.0):
        """Binarize scores at different thresholds
        """
        thresholds = np.linspace(start, 1.0, n_thresholds)
        Y_score = np.tile(y_score, (n_thresholds, 1))
        Y_pred = (Y_score >= thresholds[:, np.newaxis]).astype(int)
        return thresholds, Y_pred


def confusion_matrix(y_true, y_pred, normalize):
    pass


@validate.argument_is_proportion('top_proportion')
def precision_at(y_true, y_score, top_proportion, ignore_nas=False):
    '''
    Calculates precision at a given proportion.
    Only supports binary classification.
    '''
    # Sort scores in descending order
    scores_sorted = np.sort(y_score)[::-1]

    # Based on the proportion, get the index to split the data
    # if value is negative, return 0
    cutoff_index = max(int(len(y_true) * top_proportion) - 1, 0)
    # Get the cutoff value
    cutoff_value = scores_sorted[cutoff_index]

    # Convert scores to binary, by comparing them with the cutoff value
    scores_binary = np.array([int(y >= cutoff_value) for y in y_score])
    # Calculate precision using sklearn function
    if ignore_nas:
        precision = __precision(y_true, scores_binary)
    else:
        precision = precision_score(y_true, scores_binary)

    return precision, cutoff_value


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


@validate.argument_is_proportion('top_proportion')
def binarize_scores_at_top_proportion(y_score, top_proportion):
    """Binary scores by sorting them and grabbing a proportion from the top
    """
    cutoff_score = cutoff_score_at_top_proportion(y_score, top_proportion)
    y_score_binary = np.array(y_score >= cutoff_score).astype(int)
    return y_score_binary


@validate.argument_is_proportion('quantile')
def binarize_scores_at_quantile(y_score, quantile):
    """Binary scores at certain quantile
    """
    cutoff_score = np.quantile(y_score, quantile)
    y_score_binary = (y_score >= cutoff_score).astype(int)
    return y_score_binary


def __precision(y_true, y_pred):
    '''
        Precision metric tolerant to unlabeled data in y_true,
        NA values are ignored for the precision calculation
    '''
    # make copies of the arrays to avoid modifying the original ones
    y_true = np.copy(y_true)
    y_pred = np.copy(y_pred)

    # precision = tp/(tp+fp)
    # True nehatives do not affect precision value, so for every missing
    # value in y_true, replace it with 0 and also replace the value
    # in y_pred with 0
    is_nan = np.isnan(y_true)
    y_true[is_nan] = 0
    y_pred[is_nan] = 0
    precision = precision_score(y_true, y_pred)
    return precision


@validate.argument_is_proportion('top_proportion')
def tp_at(y_true, y_score, top_proportion):
    y_pred = binarize_scores_at_top_proportion(y_score, top_proportion)
    tp = (y_pred == 1) & (y_true == 1)
    return tp.sum()


@validate.argument_is_proportion('top_proportion')
def fp_at(y_true, y_score, top_proportion):
    y_pred = binarize_scores_at_top_proportion(y_score, top_proportion)
    fp = (y_pred == 1) & (y_true == 0)
    return fp.sum()


@validate.argument_is_proportion('top_proportion')
def tn_at(y_true, y_score, top_proportion):
    y_pred = binarize_scores_at_top_proportion(y_score, top_proportion)
    tn = (y_pred == 0) & (y_true == 0)
    return tn.sum()


@validate.argument_is_proportion('top_proportion')
def fn_at(y_true, y_score, top_proportion):
    y_pred = binarize_scores_at_top_proportion(y_score, top_proportion)
    fn = (y_pred == 0) & (y_true == 1)
    return fn.sum()


@validate.argument_is_proportion('top_proportion')
def labels_at(y_true, y_score, top_proportion, normalize=False):
    '''
        Return the number of labels encountered in the top  X proportion
    '''
    # Get indexes of scores sorted in descending order
    indexes = np.argsort(y_score)[::-1]

    # Sort true values in the same order
    y_true_sorted = y_true[indexes]

    # Grab top x proportion of true values
    cutoff_index = max(int(len(y_true_sorted) * top_proportion) - 1, 0)
    # add one to index to grab values including that index
    y_true_top = y_true_sorted[:cutoff_index+1]

    # Count the number of non-nas in the top x proportion
    # we are returning a count so it should be an int
    values = int((~np.isnan(y_true_top)).sum())

    if normalize:
        values = float(values)/(~np.isnan(y_true)).sum()

    return values
