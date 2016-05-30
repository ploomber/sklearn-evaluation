import numpy as np
from sklearn.metrics import precision_score
from . import validate


@validate.proportion
def precision_at(y_true, y_score, proportion, ignore_nas=False):
    '''
    Calculates precision at a given proportion.
    Only supports binary classification.
    '''
    # Sort scores in descending order
    scores_sorted = np.sort(y_score)[::-1]

    # Based on the proportion, get the index to split the data
    # if value is negative, return 0
    cutoff_index = max(int(len(y_true) * proportion) - 1, 0)
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


@validate.proportion
def __threshold_at(y_score, proportion):
    # Sort scores in descending order
    scores_sorted = np.sort(y_score)[::-1]
    # Based on the proportion, get the index to split th
    # if value is negative, return 0
    threshold_index = max(int(len(y_score) * proportion) - 1, 0)
    # Get the cutoff value
    threshold_value = scores_sorted[threshold_index]
    return threshold_value


@validate.proportion
def __binarize_scores_at(y_score, proportion):
    threshold_value = __threshold_at(y_score, proportion)
    y_score_binary = np.array([int(y >= threshold_value) for y in y_score])
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


@validate.proportion
def tp_at(y_true, y_score, proportion):
    y_pred = __binarize_scores_at(y_score, proportion)
    tp = (y_pred == 1) & (y_true == 1)
    return tp.sum()


@validate.proportion
def fp_at(y_true, y_score, proportion):
    y_pred = __binarize_scores_at(y_score, proportion)
    fp = (y_pred == 1) & (y_true == 0)
    return fp.sum()


@validate.proportion
def tn_at(y_true, y_score, proportion):
    y_pred = __binarize_scores_at(y_score, proportion)
    tn = (y_pred == 0) & (y_true == 0)
    return tn.sum()


@validate.proportion
def fn_at(y_true, y_score, proportion):
    y_pred = __binarize_scores_at(y_score, proportion)
    fn = (y_pred == 0) & (y_true == 1)
    return fn.sum()


@validate.proportion
def labels_at(y_true, y_score, proportion, normalize=False):
    '''
        Return the number of labels encountered in the top  X proportion
    '''
    # Get indexes of scores sorted in descending order
    indexes = np.argsort(y_score)[::-1]

    # Sort true values in the same order
    y_true_sorted = y_true[indexes]

    # Grab top x proportion of true values
    cutoff_index = max(int(len(y_true_sorted) * proportion) - 1, 0)
    # add one to index to grab values including that index
    y_true_top = y_true_sorted[:cutoff_index+1]

    # Count the number of non-nas in the top x proportion
    # we are returning a count so it should be an int
    values = int((~np.isnan(y_true_top)).sum())

    if normalize:
        values = float(values)/(~np.isnan(y_true)).sum()

    return values
