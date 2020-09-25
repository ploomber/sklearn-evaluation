from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn_evaluation.training import DataSelector
from sklearn_evaluation.training.selector import expand_arguments, _with_prefix
from sklearn_evaluation.exceptions import DataSelectorError


def _make_arr(na=0):
    arr = np.random.rand(10)
    arr[0:na] = np.nan
    return arr


def test_expand_arguments(tmp_directory, add_current_to_sys_path):
    @expand_arguments
    def func(a):
        return a

    Path('functions.py').write_text('def a_function():\n    return [1, 2, 3]')

    assert func('functions.a_function') == [1, 2, 3]
    assert func(a='functions.a_function') == [1, 2, 3]


def test_union_over():
    """Test the vectorized function
    """
    df = pd.DataFrame({'a': [], 'aa': [], 'b': [], 'c': []})

    assert set(_with_prefix(df, ['a', 'b'])) == {'a', 'aa', 'b'}


@pytest.mark.parametrize('spec, expected', [
    [{
        'names': ['x', 'y', 'z'],
    }, {'prefix_a', 'prefix_b', 'a_suffix', 'b_suffix'}],
    [{
        'prefix': 'prefix',
    }, {'x', 'y', 'z', 'a_suffix', 'b_suffix'}],
    [{
        'suffix': 'suffix',
    }, {'x', 'y', 'z', 'prefix_a', 'prefix_b'}],
    [{
        'max_na_prop': 0.5,
    }, {'x', 'y', 'z', 'prefix_a', 'prefix_b'}],
    [{
        'names': ['x'],
        'prefix': 'prefix',
        'suffix': 'suffix'
    }, {'y', 'z'}],
    [{
        'contains': 'x',
    }, {'y', 'z'}],
])
def test_column_drop(spec, expected):
    df = pd.DataFrame({
        'x': _make_arr(na=1),
        'y': _make_arr(na=2),
        'z': _make_arr(na=3),
        'prefix_a': _make_arr(na=4),
        'prefix_b': _make_arr(na=5),
        'a_suffix': _make_arr(na=6),
        'b_suffix': _make_arr(na=7),
    })

    selector = DataSelector({**spec, 'kind': 'column_drop'})
    out, _ = selector.transform(df, return_summary=True)

    assert set(out.columns) == expected


@pytest.mark.parametrize('spec, expected', [
    [{
        'if_nas': True,
    }, {1, 2}],
    [{
        'query': 'y > 10',
    }, {0}],
])
def test_row_drop(spec, expected):
    df = pd.DataFrame({
        'index': [0, 1, 2],
        'x': [np.nan, 2, 3],
        'y': [10, 20, 30],
    }).set_index('index')

    selector = DataSelector({**spec, 'kind': 'row_drop'})
    out, _ = selector.transform(df, return_summary=True)

    assert set(out.index) == expected


def test_column_keep(tmp_directory, add_current_to_sys_path):
    df = pd.DataFrame({
        'index': [0, 1, 2],
        'x': [np.nan, 2, 3],
        'y': [10, 20, 30],
    })

    Path('col_keep.py').write_text('def fn(df):\n    return ["y"]')

    selector = DataSelector({
        'kind': 'column_keep',
        'names': ['x'],
        'dotted_path': 'col_keep.fn'
    })

    out, _ = selector.transform(df, return_summary=True)

    assert set(out.columns) == {'x', 'y'}


def test_multi_step():
    df = pd.DataFrame({
        'index': [0, 1, 2],
        'x': [np.nan, 2, 3],
        'y': [10, 20, 30],
    })

    selector = DataSelector(
        {
            'kind': 'column_keep',
            'names': ['x']
        },
        {
            'kind': 'row_drop',
            'if_nas': True
        },
        {
            'kind': 'column_drop',
            'max_na_prop': 0
        },
    )

    out, _ = selector.transform(df, return_summary=True)
    assert set(out.columns) == {'x'}
    assert set(out.index) == {1, 2}


def test_repr():
    df = pd.DataFrame({'x': [1, 2, 3]})
    selector = DataSelector({'kind': 'column_drop', 'names': ['x']})
    _, summary = selector.transform(df, return_summary=True)

    assert selector._repr_html_()
    assert summary._repr_html_()


@pytest.mark.parametrize('return_summary', [True, False])
def test_error_dropping_unknown_column(return_summary):
    df = pd.DataFrame({'x': [1, 2, 3]})

    selector = DataSelector({'kind': 'column_drop', 'names': ['y']})

    with pytest.raises(DataSelectorError):
        selector.transform(df, return_summary=return_summary)
