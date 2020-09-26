from pathlib import Path
from ploomber import DAG
from ploomber.tasks import NotebookRunner
from ploomber.products import File


def config_init(app, config):
    """Note: this is executed with docs/ as working directory
    """
    dag = DAG()

    NotebookRunner(Path('source/nbs/SQLiteTracker.md'),
                   File('source/user_guide/SQLiteTracker.ipynb'),
                   dag=dag,
                   kernelspec_name='python3')

    dag.build()