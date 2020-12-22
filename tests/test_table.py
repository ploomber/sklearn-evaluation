from unittest.mock import Mock

import numpy as np
from sklearn_evaluation import table


def test_feature_importances():
    model = Mock()
    model.feature_importances_ = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    with open('tests/static/table.txt', 'r') as f:
        expected = f.read()
    assert expected == str(table.feature_importances(model))


def test_feature_importances_from_array():
    feature_importances = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    with open('tests/static/table.txt', 'r') as f:
        expected = f.read()
    assert expected == str(table.feature_importances(feature_importances))


def test_feature_importances_top3():
    ft_imp = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    with open('tests/static/table_top3.txt', 'r') as f:
        expected = f.read()
    assert expected == str(table.feature_importances(ft_imp, top_n=3))


def test_feature_importances_feature_names():
    feature_importances = np.array([0.12, 0.10, 0.8, 0.06, 0.03])
    feature_names = ['thing_a', 'thing_b', 'thing_c', 'thing_d', 'thing_e']
    with open('tests/static/table_ft_names.txt', 'r') as f:
        expected = f.read()
    assert expected == str(
        table.feature_importances(feature_importances,
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
    with open('tests/static/table_w_subestimator.txt', 'r') as f:
        expected = f.read()
    assert expected == str(table.feature_importances(rf))


def test_extend_to():
    assert table.extend_to([1], n=3) == [1, None, None]


def test_fixed_length_lists():
    assert table.fixed_length_lists([
        [1, 2, 3],
        [],
        [1],
        [1, 2],
    ]) == [[1, 2, 3], [None, None, None], [1, None, None], [1, 2, None]]


def test_from_columns():
    t = table.Table.from_columns(content=[[1, 2, 3], [4], [5, 6]],
                                 header=['a', 'b', 'c'])

    assert str(t) == ('+-----+-----+-----+'
                      '\n|   a |   b |   c |'
                      '\n+=====+=====+=====+'
                      '\n|   1 |   4 |   5 |'
                      '\n+-----+-----+-----+'
                      '\n|   2 |     |   6 |'
                      '\n+-----+-----+-----+'
                      '\n|   3 |     |     |'
                      '\n+-----+-----+-----+')
