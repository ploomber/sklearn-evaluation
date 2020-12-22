import jupytext
import nbformat
import papermill as pm
import pytest


def save_and_execute_notebook(nb_str, path):
    nb = jupytext.reads(nb_str, fmt='py:light')
    nb.metadata['kernelspec'] = {
        'name': 'python3',
        'language': 'python',
        'display_name': 'Python 3'
    }

    nbformat.write(nb, path)
    pm.execute_notebook(str(path), str(path))


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
    save_and_execute_notebook(content, 'nb_literals.ipynb')


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
    save_and_execute_notebook(content, 'nb_other_literals.ipynb')


@pytest.fixture
def nb_plot():
    content = """
import matplotlib.pyplot as plt

# + tags=["a"]
plt.plot([1, 2, 3], [1, 2, 3])

# + tags=["b"]
42
"""
    save_and_execute_notebook(content, 'nb_plot.ipynb')


@pytest.fixture
def nb_table():
    content = """
import pandas as pd

# + tags=["a"]
pd.DataFrame({'a': [1,2 ,3]})

# + tags=["b"]
42
"""
    save_and_execute_notebook(content, 'nb_table.ipynb')


@pytest.fixture
def nb_no_output():
    content = """
import pandas as pd

# + tags=["a"]
x = 1
"""
    save_and_execute_notebook(content, 'nb_no_output.ipynb')
