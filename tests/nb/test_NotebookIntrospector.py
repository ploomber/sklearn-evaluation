from sklearn_evaluation import NotebookIntrospector
from IPython.display import HTML, Image


def test_simple_notebook(tmp_directory, nb_literals):
    intr = NotebookIntrospector('nb_literals.ipynb')

    assert set(intr) == {'a', 'b', 'c'}
    assert intr['a'] == 1
    assert intr['b'] == [1, 2, 3]
    assert intr['c'] == {'x': 1, 'y': 2}
    assert dict(intr) == {'a': 1, 'b': [1, 2, 3], 'c': {'x': 1, 'y': 2}}


def test_notebook_with_plot(tmp_directory, nb_plot):
    intr = NotebookIntrospector('nb_plot.ipynb')
    img = intr['a']
    assert isinstance(img, Image)
    assert dict(intr) == {'b': 42, 'a': img}


def test_notebook_with_table(tmp_directory, nb_table):
    intr = NotebookIntrospector('nb_table.ipynb')
    html = intr['a']
    assert isinstance(html, HTML)
    assert dict(intr) == {'b': 42, 'a': html}


def test_notebook_with_no_output(tmp_directory, nb_no_output):
    intr = NotebookIntrospector('nb_no_output.ipynb')
    assert dict(intr) == dict()


def test_json_serializable(tmp_directory, nb_plot):
    d = NotebookIntrospector('nb_plot.ipynb').to_json_serializable()

    # must return the plain text representation of the plot (i.e. a str with
    # "<Figure size YxZ with 1 Axes>")
    assert 'Figure size' in d['a']
    assert d['b'] == 42
