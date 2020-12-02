import pandas as pd

from sklearn_evaluation import NotebookCollection


def test_collection(tmp_directory, nb_literals, nb_other_literals):
    col = NotebookCollection(['nb_literals.ipynb', 'nb_other_literals.ipynb'])

    assert col['a'] == [1, 2]
    assert col['b'] == [[1, 2, 3], [2, 3, 4]]
    assert col['c'] == [{'x': 1, 'y': 2}, {'x': 2, 'y': 3}]


def test_collection_to_df(tmp_directory, nb_literals, nb_other_literals):
    paths = ['nb_literals.ipynb', 'nb_other_literals.ipynb']
    col = NotebookCollection(paths, to_df=True)

    a_expected = pd.DataFrame([1, 2], index=paths)
    assert col['a'].equals(a_expected)

    b_expected = pd.DataFrame([[1, 2, 3], [2, 3, 4]], index=paths)
    assert col['b'].equals(b_expected)

    c_expected = pd.DataFrame([{
        'x': 1,
        'y': 2
    }, {
        'x': 2,
        'y': 3
    }],
                              index=paths)
    assert col['c'].equals(c_expected)
