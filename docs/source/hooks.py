from os import environ
from pathlib import Path
import urllib

import jupytext
import nbformat
from ploomber import DAG
from ploomber.tasks import NotebookRunner
from ploomber.products import File
from ploomber.constants import TaskStatus


def binder_badge(path):
    _binder_badge = """
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edublancas/sklearn-evaluation/master?filepath={})
"""
    return _binder_badge.format(urllib.parse.quote_plus(f'docs/source/{path}'))


def make_task(dag, rel_path_in, rel_path_out, base_path):
    path_in = base_path / rel_path_in
    path_out = base_path / rel_path_out

    nb = jupytext.read(path_in)

    fmt = nbformat.versions[nbformat.current_nbformat]
    nb.cells.append(fmt.new_code_cell(metadata=dict(tags=['parameters'])))

    nb.cells.insert(0, fmt.new_markdown_cell(binder_badge(rel_path_in)))

    name = Path(path_in).name.split('.')[0]
    path_preprocessed = Path(path_in).parent / (name + '-preprocessed.ipynb')
    nbformat.write(nb, path_preprocessed)

    NotebookRunner(Path(path_preprocessed),
                   File(path_out),
                   dag,
                   kernelspec_name='python3',
                   name=name,
                   local_execution=True,
                   static_analysis='disable')


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

    make_task(dag, 'nbs/SQLiteTracker.md', 'user_guide/SQLiteTracker.ipynb',
              base_path)

    make_task(dag, 'nbs/NotebookDatabase.md',
              'user_guide/NotebookDatabase.ipynb', base_path)

    make_task(dag, 'nbs/NotebookCollection.py',
              'user_guide/NotebookCollection.ipynb', base_path)

    dag.build()

    for task_name in dag:
        task = dag[task_name]

        if task.exec_status == TaskStatus.Executed:
            post_process_nb(str(task.product))
