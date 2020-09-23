import pytest
import numpy as np
import pandas as pd
from sklearn_evaluation.training import DataSelector


def _make_arr(na=0):
    arr = np.random.rand(10)
    arr[0:na] = np.nan
    return arr


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

    selector = DataSelector([('column_drop', spec)])
    out, _ = selector.fit_transform(df, return_summary=True)

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

    selector = DataSelector([('row_drop', spec)])
    out, _ = selector.fit_transform(df, return_summary=True)

    assert set(out.index) == expected


def test_row_keep():
    df = pd.DataFrame({
        'index': [0, 1, 2],
        'x': [np.nan, 2, 3],
        'y': [10, 20, 30],
    })

    selector = DataSelector([('row_keep', {'keep': ['x']})])
    out, _ = selector.fit_transform(df, return_summary=True)

    assert set(out.columns) == {'x'}
