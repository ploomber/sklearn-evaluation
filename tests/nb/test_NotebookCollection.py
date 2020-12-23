import pytest

from sklearn_evaluation import NotebookCollection
from sklearn_evaluation.nb.NotebookCollection import HTMLMapping
from IPython.display import HTML


def test_collection(tmp_directory, nb_literals, nb_other_literals):
    col = NotebookCollection(['nb_literals.ipynb', 'nb_other_literals.ipynb'],
                             ids=['nb1', 'nb2'])

    tags = ['int', 'list', 'dict']
    assert len(col) == 3
    assert list(col) == tags
    assert col._ipython_key_completions_() == tags

    assert dict(col['int']) == {'nb1': 1, 'nb2': 2}

    d = dict(col['list'])
    table = d.pop('Compare')
    assert d == {'nb1': [1, 2, 3], 'nb2': [2, 3, 4]}
    assert table.content == [(2, 1, 4), (3, None, None)]

    assert col['dict']['nb1'] == {'x': 1, 'y': 2}
    assert col['dict']['nb2'] == {'x': 2, 'y': 3}
    assert isinstance(col['dict']['Compare'], HTML)


@pytest.mark.parametrize('arg, expected', [
    [
        'filenames',
        ['nb_literals', 'nb_other_literals'],
    ],
    [
        ['a', 'b'],
        ['a', 'b'],
    ],
])
def test_custom_keys(tmp_directory, nb_literals, nb_other_literals, arg,
                     expected):
    paths = ['nb_literals.ipynb', 'nb_other_literals.ipynb']
    col = NotebookCollection(paths, ids=arg)
    assert list(col.nbs) == expected


def test_html_mapping():
    d = dict(a=1, b=2)
    m = HTMLMapping(d, 'some html string')

    assert list(m._ipython_key_completions_()) == ['a', 'b']
    assert list(m) == ['a', 'b']
    assert len(m) == 2
    assert m._repr_html_() == 'some html string'
    assert dict(m) == d
