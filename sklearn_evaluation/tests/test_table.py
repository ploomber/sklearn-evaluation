import numpy as np

try:
    from unittest.mock import Mock  # py3
except:
    from mock import Mock  # py2

from sklearn_evaluation import table


def test_feature_importances():
    model = Mock()
    model.feature_importances_ = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    with open('static/table.txt', 'r') as f:
        expected = f.read()
    assert expected == str(table.feature_importances(model))


def test_feature_importances_from_array():
    feature_importances = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    with open('static/table.txt', 'r') as f:
        expected = f.read()
    assert expected == str(table.feature_importances(feature_importances))


def test_feature_importances_top3():
    ft_imp = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    with open('static/table_top3.txt', 'r') as f:
        expected = f.read()
    assert expected == str(table.feature_importances(ft_imp, top_n=3))


def test_feature_importances_feature_names():
    feature_importances = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    feature_names = ['thing_a', 'thing_b', 'thing_c', 'thing_d', 'thing_e']
    with open('static/table_ft_names.txt', 'r') as f:
        expected = f.read()
    assert expected == str(table.feature_importances(feature_importances,
                           feature_names=feature_names))


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
    with open('static/table_w_subestimator.txt', 'r') as f:
        expected = f.read()
    assert expected == str(table.feature_importances(rf))
