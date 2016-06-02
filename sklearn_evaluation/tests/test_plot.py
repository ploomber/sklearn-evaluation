import numpy as np
from matplotlib.testing.decorators import image_comparison

from sklearn_evaluation import plot

try:
    from unittest.mock import Mock  # py3
except:
    from mock import Mock  # py2


target_names = range(2)
feature_names = range(4)
feature_importances = np.array([0.5, 0.4, 0.3, 0.2])
y_test = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
y_pred = np.array([1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
y_score = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.1, 0.1, 0.3]]).T
y_score_vector = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.1, 0.1, 0.3])


@image_comparison(baseline_images=['confusion_matrix'],
                  extensions=['png'],
                  remove_text=True)
def test_confusion_matrix():
    plot.confusion_matrix(y_test, y_pred, target_names)


@image_comparison(baseline_images=['normalized_confusion_matrix'],
                  extensions=['png'],
                  remove_text=True)
def test_normalized_confusion_matrix():
    plot.confusion_matrix(y_test, y_pred, target_names, normalize=True)


@image_comparison(baseline_images=['roc'],
                  extensions=['png'],
                  remove_text=True)
def test_roc():
    plot.roc(y_test, y_score)


@image_comparison(baseline_images=['roc'],
                  extensions=['png'],
                  remove_text=True)
def test_roc_y_score_vector():
    plot.roc(y_test, y_score_vector)


@image_comparison(baseline_images=['precision_recall'],
                  extensions=['png'],
                  remove_text=True)
def test_precision_recall():
    plot.precision_recall(y_test, y_score)


@image_comparison(baseline_images=['precision_recall'],
                  extensions=['png'],
                  remove_text=True)
def test_precision_recall_y_score_vector():
    plot.precision_recall(y_test, y_score_vector)


@image_comparison(baseline_images=['feature_importances'],
                  extensions=['png'],
                  remove_text=True)
def test_feature_importances():
    model = Mock()
    model.feature_importances_ = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    plot.feature_importances(model)


@image_comparison(baseline_images=['feature_importances'],
                  extensions=['png'],
                  remove_text=True)
def test_feature_importances_from_array():
    feature_importances = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    plot.feature_importances(feature_importances)


@image_comparison(baseline_images=['feature_importances_top3'],
                  extensions=['png'],
                  remove_text=True)
def test_feature_importances_top3():
    feature_importances = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    plot.feature_importances(feature_importances, top_n=3)


@image_comparison(baseline_images=['feature_importances_ft_names'],
                  extensions=['png'],
                  remove_text=False)
def test_feature_importances_feature_names():
    feature_importances = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    feature_names = ['thing_a', 'thing_b', 'thing_c', 'thing_d', 'thing_e']
    plot.feature_importances(feature_importances, feature_names=feature_names)


@image_comparison(baseline_images=['feature_importances_w_subestimators'],
                  extensions=['png'],
                  remove_text=True)
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


@image_comparison(baseline_images=['precision_at_proportions'],
                  extensions=['png'],
                  remove_text=True)
def test_precision_at_proportions():
    plot.precision_at_proportions(y_test, y_score)


@image_comparison(baseline_images=['precision_at_proportions'],
                  extensions=['png'],
                  remove_text=True)
def test_precision_at_proportions_y_score_vector():
    plot.precision_at_proportions(y_test, y_score_vector)
