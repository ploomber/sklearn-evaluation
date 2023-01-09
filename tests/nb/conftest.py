from pathlib import Path

import jupytext
import nbformat
import papermill as pm
import pytest


def save_notebook(nb_str, path, execute=True, parameters=None):
    nb = jupytext.reads(nb_str, fmt="py:light")
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "language": "python",
        "display_name": "Python 3",
    }

    Path(path).parent.mkdir(exist_ok=True, parents=True)

    nbformat.write(nb, path)

    if execute:
        pm.execute_notebook(str(path), str(path), parameters=parameters or {})

    return str(path)


@pytest.fixture
def nb_literals():
    content = """
# + tags=["int"]
int_ = 1
print(int_)

# + tags=["list"]
list_ = [1, 2, 3]
print(list_)

# + tags=["dict"]
dict_ = {'x': 1, 'y': 2}
dict_
"""
    save_notebook(content, "nb_literals.ipynb")


@pytest.fixture
def nb_other_literals():
    content = """
# + tags=["int"]
int_ = 2
print(int_)

# + tags=["list"]
list_ = [2, 3, 4]
print(list_)

# + tags=["dict"]
dict_ = {'x': 2, 'y': 3}
dict_
"""
    save_notebook(content, "nb_other_literals.ipynb")


@pytest.fixture
def nb_plot():
    content = """
import matplotlib.pyplot as plt

# + tags=["a"]
plt.plot([1, 2, 3], [1, 2, 3])

# + tags=["b"]
42
"""
    save_notebook(content, "nb_plot.ipynb")


@pytest.fixture
def nb_table():
    content = """
import pandas as pd

# + tags=["a"]
pd.DataFrame({'a': [1,2 ,3]})

# + tags=["b"]
42
"""
    save_notebook(content, "nb_table.ipynb")


@pytest.fixture
def nb_no_output():
    content = """
import pandas as pd

# + tags=["a"]
x = 1
"""
    save_notebook(content, "nb_no_output.ipynb")


@pytest.fixture
def nb_invalid_output():
    content = """
import numpy as np

# + tags=["numpy_array"]
np.array([1, 2, 3])
"""
    return save_notebook(content, "nb_invalid_output.ipynb")


@pytest.fixture
def nb_injected_parameters():
    content = """
import numpy as np

# + tags=["injected-parameters"]
# Parameters
x = 1
y = [1, 2]
z = {'a': 1, 'b': 2}
"""
    return save_notebook(content, "nb.ipynb", execute=False)


@pytest.fixture
def nb_injected_parameters_multiple_lines():
    content = """
import numpy as np

# + tags=["injected-parameters"]
# Parameters
x = 1
y = [1,
    2,]
z = {'a': 1,
     'b': 2}
"""
    return save_notebook(content, "nb.ipynb", execute=False)


@pytest.fixture
def nb_str():
    content = """
# + tags=["str"]
print('something')
"""
    return save_notebook(content, "nb.ipynb")


@pytest.fixture
def nb_many_str():
    content = """
# + tags=["str"]
print('a')
print('b')
print('c')
"""
    return save_notebook(content, "nb.ipynb")


@pytest.fixture
def nb_stderr():
    content = """
import sys

# + tags=["str"]
print('this should not appear', file=sys.stderr)
print('something')
"""
    return save_notebook(content, "nb.ipynb")
