from os import environ
from pathlib import Path

import jupytext
import nbformat
from ploomber import DAG
from ploomber.tasks import NotebookRunner
from ploomber.products import File
from ploomber.constants import TaskStatus


def make_task(dag, path_in, path_out):
    nb = jupytext.read(path_in)

    fmt = nbformat.versions[nbformat.current_nbformat]
    nb.cells.append(fmt.new_code_cell(metadata=dict(tags=['parameters'])))

    name = Path(path_in).name.split('.')[0]
    path_preprocessed = Path(path_in).parent / (name + '-preprocessed.ipynb')
    nbformat.write(nb, path_preprocessed)

    NotebookRunner(Path(path_preprocessed),
                   File(path_out),
                   dag,
                   kernelspec_name='python3',
                   name=name,
                   local_execution=True)


def remove_with_tag(nb, tag):
    idx = None

    for i, cell in enumerate(nb.cells):
        if tag in cell.metadata.tags:
            idx = i
            break

    nb.cells.pop(idx)


def post_process_nb(path):
    nb = jupytext.read(path)

    remove_with_tag(nb, 'injected-parameters')
    remove_with_tag(nb, 'parameters')

    jupytext.write(nb, path)


def config_init(app, config):
    """
    Note: this is executed from the docs/ directory locally (that's where the
    Makefile is) but docs/source in readthedocs.org
    """

    base_path = Path('' if environ.get('READTHEDOCS') else 'source')

    dag = DAG()

    make_task(dag, base_path / 'nbs/SQLiteTracker.md',
              base_path / 'user_guide/SQLiteTracker.ipynb')

    make_task(dag, base_path / 'nbs/NotebookCollection.py',
              base_path / 'user_guide/NotebookCollection.ipynb')

    dag.build()

    for task_name in dag:
        task = dag[task_name]

        if task.exec_status == TaskStatus.Executed:
            post_process_nb(str(task.product))
