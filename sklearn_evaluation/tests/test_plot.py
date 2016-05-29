import numpy as np
from matplotlib.testing.decorators import image_comparison, cleanup

from sklearn_evaluation import plot


target_names = range(2)
feature_names = range(4)
feature_importances = np.array([0.5, 0.4, 0.3, 0.2])
y_test = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
y_pred = np.array([1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
y_score = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.1, 0.1, 0.3]]).T
y_score_vector = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.1, 0.1, 0.3])


@cleanup
@image_comparison(baseline_images=['confusion_matrix'],
                  extensions=['png'],
                  remove_text=True)
def test_confusion_matrix():
    plot.confusion_matrix(y_test, y_pred, target_names)


@cleanup
@image_comparison(baseline_images=['normalized_confusion_matrix'],
                  extensions=['png'],
                  remove_text=True)
def test_normalized_confusion_matrix():
    plot.confusion_matrix(y_test, y_pred, target_names, normalize=True)


@cleanup
@image_comparison(baseline_images=['roc'],
                  extensions=['png'],
                  remove_text=True)
def test_roc():
    plot.roc(y_test, y_score)


@cleanup
@image_comparison(baseline_images=['roc'],
                  extensions=['png'],
                  remove_text=True)
def test_roc_y_score_vector():
    plot.roc(y_test, y_score_vector)


@cleanup
@image_comparison(baseline_images=['precision_recall'],
                  extensions=['png'],
                  remove_text=True)
def test_precision_recall():
    plot.precision_recall(y_test, y_score)


@cleanup
@image_comparison(baseline_images=['precision_recall'],
                  extensions=['png'],
                  remove_text=True)
def test_precision_recall_y_score_vector():
    plot.precision_recall(y_test, y_score_vector)

# @cleanup
# @image_comparison(baseline_images=['feature_importances'],
#                   extensions=['png'],
#                   remove_text=True)
# def test_feature_importances():
#     plot.feature_importances(model)


@cleanup
@image_comparison(baseline_images=['precision_at_proportions'],
                  extensions=['png'],
                  remove_text=True)
def test_precision_at_proportions():
    plot.precision_at_proportions(y_test, y_score)


@cleanup
@image_comparison(baseline_images=['precision_at_proportions'],
                  extensions=['png'],
                  remove_text=True)
def test_precision_at_proportions_y_score_vector():
    plot.precision_at_proportions(y_test, y_score_vector)
