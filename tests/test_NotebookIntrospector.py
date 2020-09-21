import papermill as pm
import nbformat
import jupytext
from pathlib import Path
from sklearn_evaluation import NotebookIntrospector


def test_get_tags(tmp_path):
    s = """
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
    nb_path = Path(tmp_path, 'nb.ipynb')

    nb = jupytext.reads(s, fmt='py:light')
    nb.metadata['kernelspec'] = {
        'name': 'python3',
        'language': 'python',
        'display_name': 'Python 3'
    }

    nbformat.write(nb, nb_path)
    pm.execute_notebook(str(nb_path), str(nb_path))

    intr = NotebookIntrospector(nb_path)

    assert intr.get_tags() == {'a', 'b', 'c'}

    assert intr.eval('a') == 1
    assert intr.eval('b') == [1, 2, 3]
    assert intr.eval('c') == {'x': 1, 'y': 2}
