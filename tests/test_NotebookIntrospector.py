from pathlib import Path

import pytest
import papermill as pm
import nbformat
import jupytext
from sklearn_evaluation import NotebookIntrospector

simple = """
# + tags=["a"]
a = 1
print(a)

# + tags=["b"]
b = [1, 2, 3]
print(b)

# + tags=["c"]
c = {'x': 1, 'y': 2}
print(c)
"""

plot = """
import matplotlib.pyplot as plt

# + tags=["a"]
a = 1
print(a)

# + tags=["b"]
plt.plot([1, 2, 3], [1, 2, 3])
"""

table = """
import pandas as pd

# + tags=["a"]
a = 1
print(a)

# + tags=["b"]
pd.DataFrame({'a': [1,2 ,3]})
"""


@pytest.mark.parametrize('nb_str', [
    simple,
])
def test_noteook_introspector(nb_str, tmp_path):
    nb_path = Path(tmp_path, 'nb.ipynb')

    nb = jupytext.reads(nb_str, fmt='py:light')
    nb.metadata['kernelspec'] = {
        'name': 'python3',
        'language': 'python',
        'display_name': 'Python 3'
    }

    nbformat.write(nb, nb_path)
    pm.execute_notebook(str(nb_path), str(nb_path))

    intr = NotebookIntrospector(nb_path)

    assert set(intr) == {'a', 'b', 'c'}
    assert intr.eval('a') == 1
    assert intr.eval('b') == [1, 2, 3]
    assert intr.eval('c') == {'x': 1, 'y': 2}
