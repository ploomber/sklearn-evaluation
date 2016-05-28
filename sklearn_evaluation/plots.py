"""Plotting functions."""
import numpy as np
import matplotlib.pyplot as plt
from .metrics import precision_at
from . import compute

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


# Confusion matrix
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def confusion_matrix(y_true, y_pred, target_names, ax=None, normalize=False,
                     cmap=plt.cm.Blues):
    """
    Plot confustion matrix.

    Parameters
    ----------
    y_true : array-like
        Correct target values (ground truth).
    y_pred : array-like
        Target predicted classes (estimador predictions).
    target_names : list
        List containing the names of the target classes
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes
    normalize : bool
        Normalize the confusion matrix
    cmap : matplotlib Colormap
        Colormap used for coloring the matrix


    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    cm = sk_confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)

    if ax is None:
        ax = plt.gca()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(target_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(target_names)

    title = 'Confusion matrix'
    if normalize:
        title += ' (normalized)'
    ax.set_title(title)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return ax


# Receiver operating characteristic (ROC)
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def roc(y_true, y_score, ax=None):
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : array-like, shape = [n_rows, n_classes]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_rows]
        Target scores (estimador predictions).
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    # Assumes all classes are present in y_true, binarizes and orders.
    # y_score MUST contain one column per class, so get the number of classes
    # except when is a binary classification
    if len(y_score.shape) == 1:
        n_classes = 2
    else:
        n_classes = y_score.shape[1]

    # Asume y_true is in binary format for now...

    # y_true can be in binarized form or not,
    # if it's not in binary format, binarize
    # binary_format = True
    # if not binary_format:

    y_true = label_binarize(y_true, classes=list(set(y_true)))

    # Now that both y_true is in the correct format, check input shape
    # Check y_true and y_score have correct shape

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    if n_classes > 2:
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(),
                                                  y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    else:
        fpr[1], tpr[1], _ = roc_curve(y_true, y_score)
        roc_auc[1] = auc(fpr[1], tpr[1])

    # Plot of a ROC curve for class 1 if binary classifier
    # Plot all classes and micro-average if multiclass classifier
    if ax is None:
        ax = plt.gca()

    if n_classes == 2:
        ax.plot(fpr[1], tpr[1], label=('ROC curve (area = {0:0.2f})'
                                       .format(roc_auc[1])))
    else:
        ax.plot(fpr["micro"], tpr["micro"],
                label=('micro-average ROC curve (area = {0:0.2f})'
                       .format(roc_auc["micro"])))
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=('ROC curve (area = {0:0.2f})'
                                           .format(roc_auc[i])))

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC')
    ax.legend(loc="lower right")
    return ax

# Receiver operating characteristic (ROC) with cross validation
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#example-model-selection-plot-roc-crossval-py


# Precision-recall
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
def precision_recall(y_true, y_score, ax=None):
    """
    Plot precision-recall curve.

    Parameters
    ----------
    y_true : array-like, shape = [n_rows]
        Correct target values (ground truth).
    y_score : array-like, shape = [n_rows, n_classes]
        Target scores (estimador predictions).
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    # (assumes binary input)
    # Assumes all classes are present in y_true, binarizes and orders.
    # y_score MUST contain one column per class, so get the number of classes
    # except when is a binary classification
    if len(y_score.shape) == 1:
        n_classes = 2
    else:
        n_classes = y_score.shape[1]

    # Asume y_true is in binary format for now...

    # y_true can be in binarized form or not,
    # if it's not in binary format, binarize
    # binary_format = True
    # if not binary_format:

    y_true = label_binarize(y_true, classes=list(set(y_true)))

    # Now that both y_true is in the correct format, check input shape
    # Check y_true and y_score have correct shape
    precision = dict()
    recall = dict()
    average_precision = dict()

    if n_classes > 2:
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                                y_score[:, i])
            average_precision[i] = average_precision_score(y_true[:, i],
                                                           y_score[:, i])
        # Compute micro-average ROC curve and ROC area
        precision["micro"], recall["micro"], _ = precision_recall_curve(
                                                    y_true.ravel(),
                                                    y_score.ravel())
        average_precision["micro"] = average_precision_score(y_true, y_score,
                                                             average="micro")
    else:
        precision[1], recall[1], _ = precision_recall_curve(y_true, y_score)
        average_precision[1] = average_precision_score(y_true, y_score)

    # Plot of a ROC curve for class 1 if binary classifier
    # Plot all classes and micro-average if multiclass classifier
    if ax is None:
        ax = plt.gca()

    if n_classes == 2:
        ax.plot(recall[1], precision[1], label='Precision-Recall curve')
    else:
        ax.plot(recall["micro"], precision["micro"],
                label=('micro-average Precision-recall curve (area = {0:0.2f})'
                       .format(average_precision["micro"])))
        for i in range(n_classes):
            ax.plot(recall[i], precision[i],
                    label=('P-R curve of class {0} (area = {1:0.2f})'
                           .format(i, average_precision[i])))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    if n_classes == 2:
        ax.set_title(('Precision-Recall curve: AUC={0:0.2f}'
                      .format(average_precision[1])))
    else:
        ax.set_title('Precision-Recall')
    ax.legend(loc="lower right")
    return ax


# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
def feature_importances(data, top_n=None, feature_names=None, ax=None):
    """
    Get and order feature importances from a scikit-learn model
    or from an array like structure.

    If data is a scikit-learn with sub-estimators (e.g. RandomForest,
     AdaBoost) the function will compute the standard deviation of each
     feature.

    Parameters
    ----------
    data : sklearn model or array-like structure
        Object to get the data from.
    top_n : int
        Only get results for the top_n features.
    feature_names : array-like
        Feature_names
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    # If no feature_names is provided, assign numbers
    res = compute.feature_importances(data, top_n, feature_names)
    # number of features returned
    n_feats = len(res)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Feature importances")

    try:
        ax.bar(range(n_feats), res.importance, yerr=res.std_, color='red',
               align="center")
    except:
        ax.bar(range(n_feats), res.importance, color='red',
               align="center")

    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(res.feature_name)
    ax.set_xlim([-1, n_feats])
    return ax


def precision_at_proportions(y_true, y_score, ax=None):
    """
    Plot precision values at different proportions.

    Parameters
    ----------
    y_true : array-like
        Correct target values (ground truth).
    y_score : array-like
        Target scores (estimador predictions).
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    if ax is None:
        ax = plt.gca()

    # Calculate points
    proportions = [0.01 * i for i in range(1, 101)]
    precs_and_cutoffs = [precision_at(y_true, y_score, p) for p in proportions]
    precs, cutoffs = zip(*precs_and_cutoffs)

    # Plot and set nice defaults for title and axis labels
    ax.plot(proportions, precs)
    ax.set_title('Precision at various proportions')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Proportion')
    return ax
