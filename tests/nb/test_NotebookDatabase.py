from pathlib import Path

import pytest

from sklearn_evaluation import NotebookDatabase
from conftest import save_notebook

nb = """
# + tags=["number"]
print(number)

# + tags=["list"]
print(list_)
"""


def test_notebookdatabase(tmp_directory):
    for i, params in enumerate([
            dict(number=1, list_=[1, 2, 3]),
            dict(number=2, list_=[3, 2, 1]),
    ]):
        save_notebook(nb, f'nb-{i}.ipynb', parameters=params)

    db = NotebookDatabase('nb.db', '*.ipynb')
    db.index()

    out = db.query("""
SELECT
    path,
    json_extract(c, '$.number') AS number,
    json_extract(c, '$.list') AS list
FROM nbs
ORDER BY 2 ASC
""")

    assert out.to_dict() == {
        'path': {
            0: 'nb-0.ipynb',
            1: 'nb-1.ipynb'
        },
        'number': {
            0: 1,
            1: 2
        },
        'list': {
            0: '[1,2,3]',
            1: '[3,2,1]'
        }
    }

    save_notebook(nb, 'nb-2.ipynb', parameters=dict(number=3, list_=[1, 1, 1]))

    db.index()

    out = db.query("""
SELECT
    path,
    json_extract(c, '$.number') AS number,
    json_extract(c, '$.list') AS list
FROM nbs
ORDER BY 2 ASC
""")

    assert out.to_dict() == {
        'path': {
            0: 'nb-0.ipynb',
            1: 'nb-1.ipynb',
            2: 'nb-2.ipynb'
        },
        'number': {
            0: 1,
            1: 2,
            2: 3
        },
        'list': {
            0: '[1,2,3]',
            1: '[3,2,1]',
            2: '[1,1,1]'
        }
    }


def test_notebookdatabase_plot(tmp_directory):
    nb = """
import matplotlib.pyplot as plt

# + tags=["plot"]
plt.plot([1, 1, 1])
"""

    save_notebook(nb, 'nb.ipynb')
    db = NotebookDatabase('nb.db', '*.ipynb')
    db.index()

    out = db.query("""
SELECT
    path,
    json_extract(c, '$.plot') AS plot
FROM nbs
""")

    assert out is not None


def test_notebookdatabase_empty_output(tmp_directory):
    nb = """
# + tags=["empty"]
x = 1
"""

    save_notebook(nb, 'nb.ipynb')
    db = NotebookDatabase('nb.db', '*.ipynb')
    db.index()

    out = db.query("""
SELECT
    path,
    json_extract(c, '$.empty') AS plot
FROM nbs
""")

    assert out is not None


def test_notebookdatabase_none_output(tmp_directory):
    nb = """
# + tags=["empty"]
print(None)
"""

    save_notebook(nb, 'nb.ipynb')
    db = NotebookDatabase('nb.db', '*.ipynb')
    db.index()

    out = db.query("""
SELECT
    path,
    json_extract(c, '$.plot') AS plot
FROM nbs
""")

    assert out is not None


@pytest.mark.parametrize(
    'content',
    [
        '{"a": "b"}',
        'not-a-notebook',
    ],
)
def test_notebookdatabase_corrupted_notebook(tmp_directory, content):
    Path('nb.ipynb').write_text(content)
    nb = """
# + tags=["number"]
print(42)
"""
    save_notebook(nb, 'another.ipynb')

    db = NotebookDatabase('nb.db', '*.ipynb')
    db.index()

    out = db.query("""
SELECT
    path,
    json_extract(c, '$.number') AS number
FROM nbs
""")

    assert out.to_dict() == {
        'number': {
            0: 42
        },
        'path': {
            0: 'another.ipynb'
        },
    }


def test_get_paths(tmp_directory):
    nb = """
# + tags=["number"]
print(42)
"""
    save_notebook(nb, 'nb.ipynb', execute=False)
    save_notebook(nb, 'path/to/nb2.ipynb', execute=False)

    db = NotebookDatabase('nb.db', '**/*.ipynb')
    db.index()

    assert db._get_paths() == {'nb.ipynb', 'path/to/nb2.ipynb'}


def test_notebook_database_update_false(tmp_directory):
    nb = """
# + tags=["number"]
print(42)
"""
    save_notebook(nb, 'nb.ipynb')

    db = NotebookDatabase('nb.db', '*.ipynb')
    db.index()

    nb = """
# + tags=["number"]
print(43)
"""
    save_notebook(nb, 'nb.ipynb')

    db.index()

    out = db.query("""
SELECT
    path,
    json_extract(c, '$.number') AS number
FROM nbs
""")

    assert out.to_dict() == {'number': {0: 42}, 'path': {0: 'nb.ipynb'}}


def test_notebook_database_update_true(tmp_directory):
    nb = """
# + tags=["number"]
print(42)
"""
    save_notebook(nb, 'nb.ipynb')

    db = NotebookDatabase('nb.db', '*.ipynb')
    db.index()

    nb = """
# + tags=["number"]
print(43)
"""
    save_notebook(nb, 'nb.ipynb')

    db.index(update=True)

    out = db.query("""
SELECT
    path,
    json_extract(c, '$.number') AS number
FROM nbs
""")

    assert out.to_dict() == {'number': {0: 43}, 'path': {0: 'nb.ipynb'}}
