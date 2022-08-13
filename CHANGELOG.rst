CHANGELOG
=========

0.6.1dev
--------
* Adds anonymous telemetry

0.6 (2022-08-11)
----------------
* Query notebooks with SQL using ``NotebookDatabase``
* Stripping output string in ``NotebookIntrospector``
* Ignoring standard error output in ``NotebookIntrospector``

0.5.9 (2022-07-04)
------------------
* Adds ``sort`` argument to ``plot.grid_search`` (#45)

0.5.8 (2022-04-16)
------------------
* Fixes an error in ``plot.grid_search`` when parameters grid has a single parameter

0.5.7 (2021-10-17)
------------------
* Adds ``NotebookIntrospector.get_injected_parameters``

0.5.6 (2021-06-26)
------------------
* Fixes error that caused grid search plot to fail when a parameter had a ``None`` value (#40)

0.5.5 (2021-03-28)
------------------
* Adds missing dependency (``IPython``), required by ``NotebookIntrospector``

0.5.4 (2020-12-28)
-------------------
* ``NotebookCollection`` displays output using HTML and tabs
* Adds links to try out the examples in binder


0.5.3 (2020-12-15)
-------------------
* ``DataSelector`` copies input steps to prevent mutating input params
* Simplifies ``NotebookInstrospector`` API and adds first implementation of ``NotebookCollection``


0.5.2 (2020-10-02)
------------------
* Adds SQLiteTracker for tracking ML experiments using a SQlite backend
* Adds NotebookIntrospector [Experimental]
* Migrates tests to ``nox``
* Adds DataSelector
* Enables testing with Python 3.8


0.5.1 (2020-09-18)
-------------------
* Drops support for Python 3.5
* Documentation migrated to Read the Docs
