from functools import partial
import sys
from unittest.mock import Mock

import numpy as np
from matplotlib.testing.decorators import image_comparison as _image_comparison

from sklearn_evaluation import plot

target_names = range(2)
feature_names = range(4)
feature_importances = np.array([0.5, 0.4, 0.3, 0.2])
y_test = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
y_pred = np.array([1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
y_score = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.1, 0.1, 0.3]]
).T
y_score_vector = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.1, 0.1, 0.3])

# older versions of Python are not compatible with the latest version of
# matplotlib, which leads to image differences. We increase the tolerance in
# such cases
image_comparison = partial(
    _image_comparison,
    tol=1.7 if sys.version_info.minor in (6, 7) else 0,
    remove_text=True,
    extensions=["png"],
)


@image_comparison(baseline_images=["confusion_matrix"])
def test_confusion_matrix():
    plot.confusion_matrix(y_test, y_pred, target_names)


@image_comparison(baseline_images=["normalized_confusion_matrix"])
def test_normalized_confusion_matrix():
    plot.confusion_matrix(y_test, y_pred, target_names, normalize=True)


@image_comparison(baseline_images=["roc"])
def test_roc():
    plot.roc(y_test, y_score)


@image_comparison(baseline_images=["roc"])
def test_roc_y_score_vector():
    plot.roc(y_test, y_score_vector)


@image_comparison(baseline_images=["roc"])
def test_roc_new_api():
    plot.ROC(y_test, y_score)


@image_comparison(baseline_images=["roc"])
def test_roc_y_score_vector_new_api():
    plot.ROC(y_test, y_score_vector)


@image_comparison(baseline_images=["roc_multi"])
def test_roc_multi(roc_multi_classification_values):
    y_test, y_score = roc_multi_classification_values

    plot.ROC(y_test, y_score)


@image_comparison(baseline_images=["roc", "roc2", "roc_add_roc"])
def test_roc_add_to_roc():
    y_test_roc2 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_score_roc2 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.3, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.2, 0.3, 0.5],
        ]
    ).T

    roc1 = plot.ROC(y_test, y_score)
    roc2 = plot.ROC(y_test_roc2, y_score_roc2)
    roc1 + roc2


@image_comparison(baseline_images=["roc", "roc"])
def test_roc_plot_with_fpr_and_tpr():
    roc = plot.ROC(y_test, y_score)
    roc = plot.ROC(None, None, tpr=roc.tpr, fpr=roc.fpr)


@image_comparison(baseline_images=["roc", "roc_multi", "roc_add_multi"])
def test_roc_add_to_multi(roc_multi_classification_values):
    y_test_roc2, y_score_roc2 = roc_multi_classification_values

    roc1 = plot.ROC(y_test, y_score)
    roc2 = plot.ROC(y_test_roc2, y_score_roc2)
    roc1 + roc2


@image_comparison(baseline_images=["roc_multi", "roc", "multi_add_roc"])
def test_roc_multi_add_to_roc(roc_multi_classification_values):
    y_test_roc1, y_score_roc1 = roc_multi_classification_values

    roc1 = plot.ROC(y_test_roc1, y_score_roc1)
    roc2 = plot.ROC(y_test, y_score)
    roc1 + roc2


@image_comparison(
    baseline_images=["roc_multi", "roc_multi_set2", "roc_multi_add_multi"]
)
def test_roc_multi_add_to_multi(
    roc_multi_classification_values, roc_multi_classification_values_set2
):
    y_test, y_score = roc_multi_classification_values
    y_test_roc2, y_score_roc2 = roc_multi_classification_values_set2

    roc1 = plot.ROC(y_test, y_score)
    roc2 = plot.ROC(y_test_roc2, y_score_roc2)
    roc1 + roc2


@image_comparison(baseline_images=["precision_recall"])
def test_precision_recall():
    plot.precision_recall(y_test, y_score)


@image_comparison(baseline_images=["precision_recall"])
def test_precision_recall_y_score_vector():
    plot.precision_recall(y_test, y_score_vector)


@image_comparison(baseline_images=["feature_importances"])
def test_feature_importances():
    model = Mock()
    model.feature_importances_ = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    plot.feature_importances(model)


@image_comparison(baseline_images=["feature_importances"])
def test_feature_importances_from_array():
    feature_importances = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    plot.feature_importances(feature_importances)


@image_comparison(baseline_images=["feature_importances_top3"])
def test_feature_importances_top3():
    feature_importances = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    plot.feature_importances(feature_importances, top_n=3)


@image_comparison(
    baseline_images=["feature_importances_ft_names"],
    extensions=["png"],
    remove_text=False,
)
def test_feature_importances_feature_names():
    feature_importances = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    feature_names = ["thing_a", "thing_b", "thing_c", "thing_d", "thing_e"]
    plot.feature_importances(feature_importances, feature_names=feature_names)


@image_comparison(baseline_images=["feature_importances_w_subestimators"])
def test_feature_importances_w_subestimators():
    rf = Mock()
    tree_1 = Mock()
    tree_1.feature_importances_ = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    tree_2 = Mock()
    tree_2.feature_importances_ = np.array([0.10, 0.10, 0.8, 0.06, 0.01])
    tree_3 = Mock()
    tree_3.feature_importances_ = np.array([0.09, 0.01, 0.9, 0.12, 0.02])
    tree_4 = Mock()
    tree_4.feature_importances_ = np.array([0.12, 0.10, 0.8, 0.06, 0.01])
    rf.estimators_ = [tree_1, tree_2, tree_3, tree_4]
    rf.feature_importances_ = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    plot.feature_importances(rf)


@image_comparison(baseline_images=["precision_at_proportions"])
def test_precision_at_proportions():
    plot.precision_at_proportions(y_test, y_score)


@image_comparison(baseline_images=["precision_at_proportions"])
def test_precision_at_proportions_y_score_vector():
    plot.precision_at_proportions(y_test, y_score_vector)


@image_comparison(baseline_images=["validation_curve"])
def test_validation_curve():
    acc_train = np.arange(0.1, 1.0, step=0.1)[:, np.newaxis]
    acc_val = acc_train - 0.5
    train_size = np.arange(100, 1000, step=100)

    plot.validation_curve(acc_train, acc_val, train_size)
