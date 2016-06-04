import numpy as np

from sklearn_evaluation.util import (_can_iterate, is_column_vector,
                                     is_row_vector, _group_by,
                                     _product, _mapping_to_tuple_pairs)

# can iterate


def test_can_terate_with_str():
    assert not _can_iterate('')


def test_can_terate_with_int():
    assert not _can_iterate(1)


def test_can_terate_with_float():
    assert not _can_iterate(1.0)


def test_can_terate_with_list():
    assert _can_iterate([])


def test_can_terate_with_tuple():
    assert _can_iterate(())


def test_can_terate_with_nparray():
    assert _can_iterate(np.array([0]))


# is row/column vector

def test_row_vector_from_list():
    assert is_row_vector(np.array([0]))


def test_row_vector_multid_array():
    assert not is_row_vector(np.random.rand(10, 10))


def test_column_vector():
    assert is_column_vector(np.random.rand(10, 1))


def test_column_vector_multid_array():
    assert not is_column_vector(np.random.rand(10, 2))


# group by


def test_group_by_all_different():
    all_diferent = [{1: 1}, {2: 2}, {3: 3}, {4: 4}, {5: 5}, {6: 6}]
    d = {
        1: [{1: 1}],
        2: [{2: 2}],
        3: [{3: 3}],
        4: [{4: 4}],
        5: [{5: 5}],
        6: [{6: 6}]
     }
    assert _group_by(all_diferent, lambda s: list(s.keys())[0]) == d


def test_group_by_group_by_key():
    common_key_values = [{1: 1, 'key': 'value'}, {2: 2, 'key': 'value'},
                         {3: 3, 'key': 'value'}, {4: 4, 'key': 'value2'},
                         {5: 5, 'key': 'value2'}, {6: 6, 'key': 'value3'}]
    d = {'value': [{1: 1, 'key': 'value'},
                   {2: 2, 'key': 'value'},
                   {3: 3, 'key': 'value'}],
         'value2': [{4: 4, 'key': 'value2'},
                    {5: 5, 'key': 'value2'}],
         'value3': [{6: 6, 'key': 'value3'}]}
    assert _group_by(common_key_values, 'key') == d


def test_group_by_using_fn():
    requires_fn = [{'k': 'v   '}, {'k': '     v   '}, {'k': 'v   '},
                   {'k': 'v2   '}, {'k': 'v2'}, {'k': ' v2     '}]
    d = {'v': [{'k': 'v   '}, {'k': '     v   '}, {'k': 'v   '}],
         'v2': [{'k': 'v2   '}, {'k': 'v2'}, {'k': ' v2     '}]}
    assert _group_by(requires_fn, lambda d: d['k'].strip()) == d

# test mapping to tuple


def test_mapping_tuple_single_kv():
    d = {'a': [1, 2, 3]}
    assert _mapping_to_tuple_pairs(d) == ((('a', 1),), (('a', 2),),
                                          (('a', 3),))


def test_mapping_tuple():
    d = {'a': [1, 2, 3], 'b': [4]}
    assert _mapping_to_tuple_pairs(d) == ((('a', 1), ('b', 4)),
                                          (('a', 2), ('b', 4)),
                                          (('a', 3), ('b', 4)))


def test_mapping_tuple_single_element():
    d = {'a': [1, 2, 3], 'b': 4}
    assert _mapping_to_tuple_pairs(d) == ((('a', 1), ('b', 4)),
                                          (('a', 2), ('b', 4)),
                                          (('a', 3), ('b', 4)))


def test_order_does_not_matter():
    a = {1: (1, 2, 3), 2: (4, (5, 6))}
    b = {2: (4, (5, 6)), 1: (1, 2, 3)}
    assert _mapping_to_tuple_pairs(a) == _mapping_to_tuple_pairs(b)


# test product


def test_empty():
    assert _product([], []) == []


def test_simple_product_lists():
    assert _product([1, 2], [3, 4]) == [(1, 3), (1, 4), (2, 3), (2, 4)]


def test_simple_product_lists_order_matters():
    assert _product([3, 4], [1, 2]) == [(3, 1), (3, 2), (4, 1), (4, 2)]


def test_simple_product_tuples():
    assert _product((1, 2), (3, 4)) == [(1, 3), (1, 4), (2, 3), (2, 4)]


def test_simple_product_np_array():
    a = np.array([1, 2])
    b = np.array([3, 4])
    assert _product(a, b) == [(1, 3), (1, 4), (2, 3), (2, 4)]


def test_single_with_list():
    assert _product(1, [2, 3]) == [(1, 2), (1, 3)]


def test_list_with_single():
    assert _product([1, 2], 3) == [(1, 3), (2, 3)]


def test_one_len_lists():
    assert _product([1], [2]) == [(1, 2)]


def test_mixed_types():
    a = ['a', 'b']
    b = [(1, 2, 3)]
    assert _product(a, b) == [('a', (1, 2, 3)), ('b', (1, 2, 3))]
