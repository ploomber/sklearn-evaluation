__version__ = '0.7.1'

from ploomber_core.telemetry.telemetry import Telemetry

from .evaluator import ClassifierEvaluator
from .nb.NotebookIntrospector import NotebookIntrospector
from .nb.NotebookCollection import NotebookCollection
from .nb.NotebookDatabase import NotebookDatabase
from .SQLiteTracker import SQLiteTracker

__all__ = [
    'ClassifierEvaluator',
    'NotebookIntrospector',
    'SQLiteTracker',
    'NotebookCollection',
    'NotebookDatabase',
]

telemetry = Telemetry(
    api_key="phc_P9SpSeypyPwxrMdFn2edOOEooQioF2axppyEeDwtMSP",
    package_name="sklearn-evaluation",
    version=__version__,
)

telemetry.log_api('imported')
