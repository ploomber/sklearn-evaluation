import pytest

from sklearn_evaluation import NotebookCollection


def test_collection(tmp_directory, nb_literals, nb_other_literals):
    col = NotebookCollection(['nb_literals.ipynb', 'nb_other_literals.ipynb'],
                             ids=['nb1', 'nb2'])
    assert dict(col['a']) == {'nb1': 1, 'nb2': 2}
    assert dict(col['b']) == {'nb1': [1, 2, 3], 'nb2': [2, 3, 4]}
    assert col['c']['nb1'] == {'x': 1, 'y': 2}
    assert col['c']['nb2'] == {'x': 2, 'y': 3}
    assert col['c']['Compare']


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
