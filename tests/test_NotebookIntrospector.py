from pathlib import Path

import papermill as pm
import nbformat
import jupytext
from sklearn_evaluation import NotebookIntrospector
from IPython.display import HTML, Image


def save_and_execute_notebook(nb_str, path):
    nb = jupytext.reads(nb_str, fmt='py:light')
    nb.metadata['kernelspec'] = {
        'name': 'python3',
        'language': 'python',
        'display_name': 'Python 3'
    }

    nbformat.write(nb, path)
    pm.execute_notebook(str(path), str(path))


def test_simple_notebook(tmp_path):
    nb_str = """
# + tags=["a"]
a = 1
print(a)

# + tags=["b"]
b = [1, 2, 3]
print(b)

# + tags=["c"]
c = {'x': 1, 'y': 2}
c
"""

    path = Path(tmp_path, 'nb.ipynb')
    save_and_execute_notebook(nb_str, path)

    intr = NotebookIntrospector(path)

    assert set(intr) == {'a', 'b', 'c'}
    assert intr.eval('a') == 1
    assert intr.eval('b') == [1, 2, 3]
    assert intr.eval('c') == {'x': 1, 'y': 2}
    assert intr.to_dict(eval_=True) == {
        'a': 1,
        'b': [1, 2, 3],
        'c': {
            'x': 1,
            'y': 2
        }
    }


def test_notebook_with_plot(tmp_path):
    nb_str = """
import matplotlib.pyplot as plt

# + tags=["a"]
plt.plot([1, 2, 3], [1, 2, 3])

# + tags=["b"]
42
"""

    path = Path(tmp_path, 'nb.ipynb')
    save_and_execute_notebook(nb_str, path)

    intr = NotebookIntrospector(path)
    assert isinstance(intr['a'], Image)
    assert intr.to_dict() == {'b': 42}


def test_notebook_with_table(tmp_path):
    nb_str = """
import pandas as pd

# + tags=["a"]
pd.DataFrame({'a': [1,2 ,3]})

# + tags=["b"]
42
"""

    path = Path(tmp_path, 'nb.ipynb')
    save_and_execute_notebook(nb_str, path)

    intr = NotebookIntrospector(path)
    assert isinstance(intr['a'], HTML)
    assert intr.to_dict() == {'b': 42}
