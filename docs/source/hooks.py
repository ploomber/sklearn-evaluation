from os import environ
from pathlib import Path
from ploomber import DAG
from ploomber.tasks import NotebookRunner
from ploomber.products import File


def config_init(app, config):
    """
    Note: this is executed from the docs/ directory locally (that's where the
    Makefile is) but docs/source in readthedocs.org
    """

    base_path = Path('' if environ.get('READTHEDOCS') else 'source')

    dag = DAG()

    NotebookRunner(base_path / 'nbs/SQLiteTracker.md',
                   File(base_path / 'user_guide/SQLiteTracker.ipynb'),
                   dag=dag,
                   kernelspec_name='python3')

    dag.build()
