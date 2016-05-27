"""Plotting functions."""
import numpy as np
import matplotlib.pyplot as plt
import tables
from metrics import precision_at

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


# Confusion matrix
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def confusion_matrix(y_true, y_pred, target_names, ax=None, normalize=False,
                     title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plot confustion matrix.

    Parameters
    ----------
    y_true : array-like
        Correct target values (Ground truth).
    y_pred : array-like
        Target predicted classes (Model predictions).
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

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
    ax.set_title(title)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return ax


# Receiver operating characteristic (ROC)
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def roc(y_true, y_score, ax=None, title="ROC curve"):
    '''
        Plot ROC curve based on true labels and model predictions.
        y_score (n_rows * n_classes) - Scores for a given prediction
        y_true  (n_rows * 1) - True label for a given prediction
        Assumes all classes are present in y_true, binarizes and orders.
    '''
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
        ax.plot(fpr[1], tpr[1], label='ROC curve (area = %0.2f)' % roc_auc[1])
    else:
        ax.plot(fpr["micro"], tpr["micro"],
                label=('micro-average ROC curve (area = {0:0.2f})'
                       .format(roc_auc["micro"])))
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=('ROC curve (area = %0.2f)'
                                           .format(roc_auc[i])))

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    return ax

# Receiver operating characteristic (ROC) with cross validation
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html


# Precision-recall
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
def precision_recall(y_true, y_score, ax=None, title="Precision-Recall curve"):
    '''
        Plot Precision-Recall curve based on true labels and model predictions.
        y_score (n_rows * n_classes) - Scores for a given prediction
        y_true  (n_rows * 1) - True label for a given prediction
         (assumes binary input)
        Assumes all classes are present in y_true, binarizes and orders.
    '''
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
        ax.set_title(title)
    ax.legend(loc="lower right")
    return ax


# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
def feature_importances(model, ax=None, feature_names=None, n=None):
    # If no feature_names is provided, assign numbers
    total = len(model.feature_importances_)
    feature_names = range(total) if feature_names is None else feature_names
    # Plot all features if n is not provided, otherwise plot top n features
    n = len(feature_names) if n is None else n
    # Compute feature importances, use private method to avoid getting
    # formatted results
    f_imp = tables._compute_feature_importances(model, feature_names)
    importances = map(lambda x: x['importance'], f_imp)[:n]
    stds = map(lambda x: x['std'], f_imp)[:n]
    names = map(lambda x: x['name'], f_imp)[:n]

    if ax is None:
        ax = plt.gca()

    ax.set_title("Feature importances")
    ax.bar(range(len(importances)), importances, color="r", yerr=stds,
           align="center")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_xlim([-1, 10])
    return ax


def feature_importances_from_list(features, feature_importances, ax=None,
                                  top_n=None):
    '''
        Plot top_n features by passing a list of features and a list of
        features importances.
    '''
    fts = zip(features, feature_importances)
    fts.sort(key=lambda t: t[1])
    top_n = top_n if top_n else len(features)
    top_fts = fts[::-1][:top_n]
    names = [ft[0] for ft in top_fts]
    importances = [ft[1] for ft in top_fts]

    if ax is None:
        ax = plt.gca()

    ax.set_title("Feature importances")
    ax.bar(range(len(importances)), importances, color="r", align="center")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_xlim([-1, 10])
    return ax


def precision_at_proportions(y_true, y_score, ax=None, **kwargs):
    """
    Plot precision values at different proportions.

    Parameters
    ----------
    y_true : array-like
        Correct target values (Ground truth).
    y_score : array-like
        Target scores (Model predictions).
    ax: matplotlib Axes
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
    ax.plot(proportions, precs, **kwargs)
    ax.set_title('Precision at various proportions')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Proportion')
    return ax
