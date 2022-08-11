from sklearn_evaluation import NotebookIntrospector
from IPython.display import HTML, Image


def test_simple_notebook(tmp_directory, nb_literals):
    intr = NotebookIntrospector('nb_literals.ipynb')

    assert set(intr) == {'int', 'list', 'dict'}
    assert intr['int'] == 1
    assert intr['list'] == [1, 2, 3]
    assert intr['dict'] == {'x': 1, 'y': 2}
    assert dict(intr) == {
        'int': 1,
        'list': [1, 2, 3],
        'dict': {
            'x': 1,
            'y': 2
        }
    }


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


def test_notebook_with_invalid_output(tmp_directory, nb_invalid_output):
    intr = NotebookIntrospector(nb_invalid_output, literal_eval=True)
    assert intr['numpy_array'] == 'array([1, 2, 3])'


def test_json_serializable(tmp_directory, nb_plot):
    d = NotebookIntrospector('nb_plot.ipynb').to_json_serializable()

    # must return the plain text representation of the plot (i.e. a str with
    # "<Figure size YxZ with 1 Axes>")
    assert 'Figure size' in d['a']
    assert d['b'] == 42


def test_get_injected_parameters(tmp_directory, nb_injected_parameters):
    d = NotebookIntrospector('nb.ipynb')

    assert d.get_injected_parameters() == {
        'x': 1,
        'y': [1, 2],
        'z': {
            'a': 1,
            'b': 2
        }
    }


def test_get_injected_parameters_multiple_lines(
        tmp_directory, nb_injected_parameters_multiple_lines):
    d = NotebookIntrospector('nb.ipynb')

    assert d.get_injected_parameters() == {
        'x': 1,
        'y': [1, 2],
        'z': {
            'a': 1,
            'b': 2
        }
    }


def test_ignores_standard_error():
    raise NotImplementedError


def test_strips_whitespace(tmp_directory, nb_str):
    d = NotebookIntrospector('nb.ipynb')
    assert d['str'] == 'something'
