import pytest
import pandas as pd

from sklearn_evaluation import NotebookCollection


def test_collection(tmp_directory, nb_literals, nb_other_literals):
    col = NotebookCollection(['nb_literals.ipynb', 'nb_other_literals.ipynb'],
                             keys=['nb1', 'nb2'])
    assert dict(col['a']) == {'nb1': 1, 'nb2': 2}
    assert dict(col['b']) == {'nb1': [1, 2, 3], 'nb2': [2, 3, 4]}
    nb1 = pd.DataFrame({'x': 1, 'y': 2}, index=[0])
    assert col['c']['nb1'].equals(nb1)
    nb2 = pd.DataFrame({'x': 2, 'y': 3}, index=[0])
    assert col['c']['nb2'].equals(nb2)

    summary = pd.DataFrame({
        'nb1': [1, 2],
        'nb2': [2, 3],
        'diff': [1, 1]
    },
                           index=['x', 'y'])

    assert col['c']['Summary'].data.equals(summary)


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
    col = NotebookCollection(paths, keys=arg)
    assert list(col.nbs) == expected
