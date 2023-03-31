import pytest

import pandas as pd
import numpy as np
from sklearn_evaluation import NotebookCollection
from sklearn_evaluation.nb import NotebookCollection as nbcollection
from IPython.display import HTML


def test_collection(tmp_directory, nb_literals, nb_other_literals):
    col = NotebookCollection(
        ["nb_literals.ipynb", "nb_other_literals.ipynb"], ids=["nb1", "nb2"]
    )

    tags = ["int", "list", "dict"]
    assert len(col) == 3
    assert list(col) == tags
    assert col._ipython_key_completions_() == tags

    assert dict(col["int"]) == {"nb1": 1, "nb2": 2}

    d = dict(col["list"])
    table = d.pop("Compare")
    assert d == {"nb1": [1, 2, 3], "nb2": [2, 3, 4]}
    assert table.content == [(2, 1, 4), (3, None, None)]

    assert col["dict"]["nb1"] == {"x": 1, "y": 2}
    assert col["dict"]["nb2"] == {"x": 2, "y": 3}
    assert isinstance(col["dict"]["Compare"], HTML)


@pytest.mark.parametrize(
    "arg, expected",
    [
        [
            "filenames",
            ["nb_literals", "nb_other_literals"],
        ],
        [
            ["a", "b"],
            ["a", "b"],
        ],
    ],
)
def test_custom_keys(tmp_directory, nb_literals, nb_other_literals, arg, expected):
    paths = ["nb_literals.ipynb", "nb_other_literals.ipynb"]
    col = NotebookCollection(paths, ids=arg)
    assert list(col.nbs) == expected


def test_html_mapping():
    d = dict(a=1, b=2)
    m = nbcollection.HTMLMapping(d, "some html string")

    assert list(m._ipython_key_completions_()) == ["a", "b"]
    assert list(m) == ["a", "b"]
    assert len(m) == 2
    assert m._repr_html_() == "some html string"
    assert dict(m) == d


def test_get_filename():
    assert nbcollection._get_filename("/a/b/c.ipynb") == "c"


@pytest.mark.parametrize(
    "elements, value, expected",
    [
        [False, "value", False],
        [True, "value", True],
        [None, "value", False],
        [["another_value"], "value", False],
        [["another_value", "value"], "value", True],
    ],
)
def test_is_in(elements, value, expected):
    assert nbcollection.is_in(elements, value) == expected


@pytest.mark.parametrize(
    "scores_arg, errors_expected, scores_expected",
    [
        [
            False,
            ([1, 2, 3], ["a", "b", "c"]),
            ([], ["a", "b", "c"]),
        ],
        [
            True,
            ([], ["a", "b", "c"]),
            ([1, 2, 3], ["a", "b", "c"]),
        ],
        [
            [1],
            ([2, 3], ["a", "b", "c"]),
            ([1], ["a", "b", "c"]),
        ],
    ],
)
@pytest.mark.parametrize("transpose", [False, True])
def test_split_errors_and_scores(
    scores_arg, errors_expected, scores_expected, transpose
):
    errors, scores = nbcollection.split_errors_and_scores(
        axis=[1, 2, 3],
        scores_arg=scores_arg,
        axis_second=["a", "b", "c"],
        transpose=transpose,
    )

    if transpose:
        scores_expected = (scores_expected[1], scores_expected[0])
        errors_expected = (errors_expected[1], errors_expected[0])

    assert errors == errors_expected
    assert scores == scores_expected


def test_data2html_img():
    assert (
        nbcollection.data2html_img(bytes([0]))
        == '<img src="data:image/png;base64, AA==\n"/>'
    )


def test_to_df_from_df():
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = nbcollection.to_df(df)

    assert df.equals(out)
    assert df.columns == out.columns
    assert df.index.name == out.index.name


def test_to_df_from_html():
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = nbcollection.to_df(HTML(df._repr_html_()))

    assert df.equals(out)
    assert df.columns == out.columns
    assert df.index.name == out.index.name


def test_to_df_from_htm_with_index_name():
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.index.name = "index"
    out = nbcollection.to_df(HTML(df._repr_html_()))

    assert df.equals(out)
    assert df.columns == out.columns
    assert df.index.name == out.index.name


@pytest.fixture
def metrics():
    m1 = pd.DataFrame({"mae": [0.1], "mse": [0.2]})
    m2 = pd.DataFrame({"mae": [0.15], "mse": [0.25]})
    m3 = pd.DataFrame({"mae": [0.25], "mse": [0.35]})
    return [m1, m2, m3]


def test_compare_df_two_metrics(metrics):
    expected = pd.DataFrame(
        {
            1: [0.1, 0.2],
            2: [0.15, 0.25],
            "diff": [0.04999999999999999, 0.04999999999999999],
            "diff_relative": [0.33333333333333326, 0.19999999999999996],
            "ratio": [1.4999999999999998, 1.25],
        }
    )

    out = nbcollection.compare_df(tables=metrics[:2], ids=[1, 2], scores_arg=None)

    np.testing.assert_allclose(out.data, expected)
    assert list(out.data.columns) == list(expected.columns)
    assert out.data.index.name == expected.index.name


def test_compare_df_three_metrics(metrics):
    expected = pd.DataFrame({1: [0.1, 0.2], 2: [0.15, 0.25], 3: [0.25, 0.35]})

    out = nbcollection.compare_df(tables=metrics, ids=[1, 2, 3], scores_arg=None)

    np.testing.assert_allclose(out.data, expected)
    assert list(out.data.columns) == list(expected.columns)
    assert out.data.index.name == expected.index.name


@pytest.fixture
def tables():
    t1 = pd.DataFrame({"mae": [0.1, 0.11], "mse": [0.2, 0.21]}).T
    t2 = pd.DataFrame({"mae": [0.15, 0.16], "mse": [0.25, 0.26]}).T
    t3 = pd.DataFrame({"mae": [0.25, 0.26], "mse": [0.35, 0.36]}).T
    return [t1, t2, t3]


def test_compare_df_two_tables(tables):
    expected = pd.DataFrame(
        {
            0: {"mae": 0.04999999999999999, "mse": 0.04999999999999999},
            1: {"mae": 0.05, "mse": 0.05000000000000002},
        }
    )

    out = nbcollection.compare_df(tables=tables[:2], ids=[1, 2], scores_arg=None)

    np.testing.assert_allclose(out.data, expected)
    assert list(out.data.columns) == list(expected.columns)
    assert out.data.index.name == expected.index.name


def test_compare_df_three_tables(tables):
    out = nbcollection.compare_df(tables=tables, ids=[1, 2, 3], scores_arg=None)

    assert out is None
