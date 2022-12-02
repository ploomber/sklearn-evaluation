from collections.abc import Mapping
from uuid import uuid4
import sqlite3
import json
import importlib

import pandas as pd
from jinja2 import Template

from sklearn_evaluation.table import Table
from sklearn_evaluation.report.serialize import try_serialize_figures, figure2html
from sklearn_evaluation.telemetry import SKLearnEvaluationLogger
from sklearn_evaluation.nb.NotebookCollection import (
    add_compare_tab,
    tabs_html_from_content,
    HTMLMapping,
)
from sklearn_evaluation import plot

minor = sqlite3.sqlite_version.split(".")[1]

# -> and ->> operators introduced in sqlite version 3.38.0
# https://www.sqlite.org/json1.html#jptr
ARROW_OPERATOR_SUPPORTED = int(minor) >= 38

TEMPLATE_ARROW = """\
SELECT
    uuid,
    {% for p, alias in keys -%}
    parameters ->> '{{p}}' as {{alias}}{% if not loop.last %},{% endif %}
    {% endfor -%}
FROM experiments
LIMIT 10
"""

TEMPLATE_JSON_EXTRACT = """\
SELECT
    uuid,
    {% for p, alias in keys -%}
    json_extract(parameters, '$.{{p}}') as {{alias}}{% if not loop.last %},{% endif %}
    {% endfor -%}
FROM experiments
LIMIT 10
"""


class Experiment:
    """An experiment instance used to log values"""

    def __init__(self, tracker, uuid, data) -> None:
        self._tracker = tracker
        self._uuid = uuid
        self._data = data

    @classmethod
    def new(cls, tracker):
        return cls(tracker=tracker, uuid=tracker.new(), data=None)

    @property
    def uuid(self):
        return self._uuid

    def log_confusion_matrix(self, y_true, y_pred, target_names=None, normalize=False):
        """Log a confusion matrix

        Examples
        --------
        >>> from sklearn_evaluation import SQLiteTracker
        >>> tracker = SQLiteTracker("experiments.db")
        >>> exp = tracker.new_experiment()
        >>> exp.log_confusion_matrix([1, 1, 0, 0], [1, 0, 1, 0]) # doctest: +SKIP
        >>> data = tracker.get(exp.uuid)
        >>> data['confusion_matrix'] # doctest: +SKIP
        """
        cm = plot.ConfusionMatrix.from_raw_data(
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names,
            normalize=normalize,
        )
        self._tracker.upsert(self._uuid, {"confusion_matrix": cm._get_data()})
        return cm

    def log_classification_report(
        self, y_true, y_pred, *, target_names=None, sample_weight=None, zero_division=0
    ):
        """Log classification report

        Examples
        --------
        >>> from sklearn_evaluation import SQLiteTracker
        >>> tracker = SQLiteTracker("experiments.db")
        >>> exp = tracker.new_experiment()
        >>> exp.log_classification_report([1, 1, 0, 0], [1, 0, 1, 0]) # doctest: +SKIP
        >>> data = tracker.get(exp.uuid)
        >>> data['classification_report'] # doctest: +SKIP
        """
        cr = plot.ClassificationReport.from_raw_data(
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

        self._tracker.upsert(self._uuid, {"classification_report": cr._get_data()})
        return cr

    def log(self, key, obj):
        """Log a value. Any JSON-serializable object works

        Examples
        --------
        >>> from sklearn_evaluation import SQLiteTracker
        >>> tracker = SQLiteTracker("experiments.db")
        >>> exp = tracker.new_experiment()
        >>> exp.log("accuracy", 0.8)
        0.8
        >>> data = tracker.get(exp.uuid)
        >>> data['accuracy']
        0.8
        """
        self._tracker.upsert(self._uuid, {key: obj})
        return obj

    def log_dict(self, obj):
        """Log a dictionary with values

        Examples
        --------
        >>> from sklearn_evaluation import SQLiteTracker
        >>> tracker = SQLiteTracker("experiments.db")
        >>> exp = tracker.new_experiment()
        >>> exp.log_dict({"precision": 0.9, "recall": 0.7})
        {'precision': 0.9, 'recall': 0.7}
        >>> data = tracker.get(exp.uuid)
        >>> data['precision']
        0.9
        >>> data['recall']
        0.7
        """
        self._tracker.upsert(self._uuid, obj)
        return obj

    def log_figure(self, key, fig):
        """Log a matplotlib figure

        >>> import matplotlib.pyplot as plt
        >>> from sklearn_evaluation import SQLiteTracker
        >>> tracker = SQLiteTracker("experiments.db")
        >>> fig, ax = plt.subplots()
        >>> ax.scatter([1, 2, 3], [1, 2, 3]) # doctest: +SKIP
        >>> exp = tracker.new_experiment()
        >>> exp.log_figure("scatter", fig)
        >>> data = tracker.get(exp.uuid)
        >>> data['scatter'] # doctest: +SKIP
        """
        self._tracker.upsert(self._uuid, {key: figure2html(fig)})

    def comment(self, comment):
        """Add a comment to an experiment

        Examples
        --------
        >>> from sklearn_evaluation import SQLiteTracker
        >>> tracker = SQLiteTracker("experiments.db")
        >>> exp = tracker.new_experiment()
        >>> exp.comment("some comment") # add comment at runtime
        >>> retrieved = tracker.get(exp.uuid)
        >>> retrieved.comment("another commment") # add comment after running the experiment
        """
        self._tracker.comment(self._uuid, comment)

    def __repr__(self) -> str:
        class_ = type(self).__name__
        data = self._data
        return f"{class_}({data!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            other = other._data

        return self._data == other

    def __getitem__(self, key):
        return self._data[key]


class SQLiteTracker:
    """A experiment tracker backed by a SQLite database

    :doc:`Click here <../user_guide/SQLiteTracker>` to see the user guide.

    Parameters
    ----------
    path
        Database location

    Examples
    --------
    >>> from sklearn_evaluation import SQLiteTracker
    >>> tracker = SQLiteTracker("experiments.db")
    >>> experiment = tracker.new_experiment() # new experiment
    >>> experiment.log("accuracy", 0.8) # log metric
    0.8
    >>> tracker.get(experiment.uuid) # retrieve it later with the uuid
    Experiment({'accuracy': 0.8})
    >>> experiment.log_confusion_matrix([1, 1, 0, 0], [1, 0, 1, 0]) # doctest: +SKIP
    >>> data = tracker.get(experiment.uuid)
    >>> data['confusion_matrix'] # doctest: +SKIP

    """

    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)

        cur = self.conn.cursor()
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS experiments (
            uuid TEXT NOT NULL UNIQUE,
            created TIMESTAMP default current_timestamp,
            parameters TEXT,
            comment TEXT
        )
        """
        )
        cur.close()

    def __getitem__(self, uuid):
        """Get experiment with a given uuid"""
        # TODO: make it work for a list of uuids
        return pd.read_sql(
            "SELECT * FROM experiments WHERE uuid = ?",
            self.conn,
            params=[uuid],
            index_col="uuid",
        )

    def recent(self, n=5, normalize=False):
        """Get most recent experiments as a pandas.DataFrame"""
        query = """
        SELECT uuid, created, parameters, comment
        FROM experiments
        ORDER BY created DESC
        LIMIT ?
        """
        df = pd.read_sql(query, self.conn, params=[n], index_col="uuid")

        def _json_loads(s):
            return {} if not s else json.loads(s)

        if normalize:
            # parse and normalize json
            parameters = pd.json_normalize(
                df.pop("parameters").apply(_json_loads)
            ).set_index(df.index)
            df = df.join(parameters)

            # re order columns to show "comment" at the end
            comment = df.pop("comment")
            df.insert(len(df.columns), "comment", comment)

        return df

    @SKLearnEvaluationLogger.log(feature="SQLiteTracker")
    def query(self, code, as_frame=True, render_plots=False):
        """Query the database

        Parameters
        ----------
        code : str
            The SQL query to execute

        as_frame : bool, default=True
            If True, it'll return the results of your query in a
            pandas.DataFrame, otherwise it'll return a Results object.
            The Results object can render HTML stored in the database but
            cannot be filtered or manipulated like a pandas.DataFrame

        render_plots: bool, default=False
            Whether to render plots in the results or not. Only valid when
            as_frame=False

        Examples
        --------
        >>> from sklearn_evaluation import SQLiteTracker
        >>> tracker = SQLiteTracker('experiments.db')
        >>> exp1 = tracker.new_experiment()
        >>> exp1.log("accuracy", 0.8) # doctest: +SKIP
        >>> exp1.log_confusion_matrix([1, 1, 0, 0], [1, 0, 1, 0]) # doctest: +SKIP
        >>> exp2 = tracker.new_experiment()
        >>> exp2.log("accuracy", 1.0) # doctest: +SKIP
        >>> exp2.log_confusion_matrix([1, 1, 0, 0], [1, 1, 0, 0]) # doctest: +SKIP

        >>> df = tracker.query('''
        ... SELECT uuid,
        ...        json_extract(parameters, '$.accuracy') AS accuracy,
        ...        json_extract(parameters, '$.confusion_matrix') AS cm
        ... FROM experiments
        ... ''', as_frame=True)


        >>> results = tracker.query('''
        ... SELECT uuid,
        ...        json_extract(parameters, '$.accuracy') AS accuracy,
        ...        json_extract(parameters, '$.confusion_matrix') AS cm
        ... FROM experiments
        ... ''', as_frame=False, render_plots=True)
        """
        if as_frame:
            df = pd.read_sql(code, self.conn)

            if "uuid" in df:
                df = df.set_index("uuid")

            return df
        else:
            cursor = self.conn.execute(code)
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            return Results(columns, rows, render_plots=render_plots)

    @SKLearnEvaluationLogger.log(feature="SQLiteTracker")
    def new(self):
        """Create a new experiment, returns a uuid"""
        uuid = str(uuid4())[:8]
        cur = self.conn.cursor()
        cur.execute(
            """
        INSERT INTO experiments (uuid)
        VALUES(?)
        """,
            [uuid],
        )
        cur.close()
        self.conn.commit()
        return uuid

    @SKLearnEvaluationLogger.log(feature="SQLiteTracker")
    def new_experiment(self):
        """Returns an experiment instance"""
        return Experiment.new(self)

    @SKLearnEvaluationLogger.log(feature="SQLiteTracker")
    def update(self, uuid, parameters, allow_overwrite=False):
        """Update the parameters of a experiment given its uuid"""
        if not allow_overwrite:
            self._can_update(uuid)

        cur = self.conn.cursor()
        cur.execute(
            """
        UPDATE experiments
        SET parameters = ?
        WHERE uuid = ?
        """,
            [json.dumps(parameters), uuid],
        )
        cur.close()
        self.conn.commit()

    def upsert(self, uuid, parameters):
        """Modify the stored parameters of an existing experiment"""
        existing = self.get(uuid, unserialize_plots=False)._data
        parameters_new = json.dumps({**existing, **parameters})

        cur = self.conn.cursor()
        cur.execute(
            """
        UPDATE experiments
        SET parameters = ?
        WHERE uuid = ?
        """,
            [parameters_new, uuid],
        )
        cur.close()
        self.conn.commit()

    @SKLearnEvaluationLogger.log("SQLiteTracker")
    def insert(self, uuid, parameters):
        """Insert a new experiment"""
        # serialize matplotlib.figure.Figure, if any
        parameters = try_serialize_figures(parameters)

        cur = self.conn.cursor()
        cur.execute(
            """
        INSERT INTO experiments (uuid, parameters)
        VALUES(?, ?)
        """,
            [uuid, json.dumps(parameters)],
        )
        cur.close()
        self.conn.commit()

    @SKLearnEvaluationLogger.log("SQLiteTracker")
    def insert_many(self, parameters_all):
        """Insert many experiments at once"""
        cur = self.conn.cursor()
        cur.execute("BEGIN TRANSACTION")

        for parameters in parameters_all:
            uuid = str(uuid4())[:8]
            cur.execute(
                """
        INSERT INTO experiments (uuid, parameters)
        VALUES(?, ?)
        """,
                [uuid, json.dumps(parameters)],
            )

        cur.close()
        self.conn.commit()

    @SKLearnEvaluationLogger.log("SQLiteTracker")
    def comment(self, uuid, comment):
        """Add a comment to an experiment given its uuid"""
        # TODO: add overwrite (false by default) and append options
        cur = self.conn.cursor()
        cur.execute(
            """
        UPDATE experiments
        SET comment = ?
        WHERE uuid = ?
        """,
            [comment, uuid],
        )
        cur.close()
        self.conn.commit()

    def _recent(self, n=5, fmt="html"):
        if fmt not in {"html", "plain"}:
            raise ValueError('fmt must be one "html" or "plain"')

        cur = self.conn.cursor()
        cur.execute(
            """
        SELECT uuid, created, parameters, comment
        FROM experiments
        ORDER BY created DESC
        LIMIT ?
        """,
            [n],
        )
        res = cur.fetchall()
        table = Table(res, header=["uuid", "created", "parameters", "comment"])

        title_template = "<h4> {} </h4>" if fmt == "html" else "{}\n"
        title = title_template.format(type(self).__name__)

        if not len(table):
            title += "(No experiments saved yet)"
            if fmt == "plain":
                title += "\n"

        if len(table):
            footer = ("<br>" if fmt == "html" else "\n") + "(Most recent experiments)"
        else:
            footer = ""

        return title + (table.to_html() if fmt == "html" else str(table)) + footer

    def _can_update(self, uuid):
        """Check if an experiment with a given uuid can be updated"""
        cur = self.conn.cursor()

        cur.execute(
            """
        SELECT parameters
        FROM experiments
        WHERE uuid = ?
        """,
            [uuid],
        )

        row = cur.fetchone()

        exists = row is not None

        if exists:
            empty = row[0] is None

            if not empty:
                raise ValueError(
                    "Cannot update non-empty experiment with " 'uuid "{}"'.format(uuid)
                )
        else:
            raise ValueError(
                "Cannot update experiment with "
                'uuid "{}" because it does '
                "not exist".format(uuid)
            )

    @SKLearnEvaluationLogger.log("SQLiteTracker")
    def get_parameters_keys(self, limit=100):
        """
        Return the keys in the parameters column by randomly sampling records
        and obtaining the keys of the JSON objects
        """
        keys = set()
        cur = self.conn.execute(
            """
        SELECT parameters
        FROM experiments
        ORDER BY RANDOM()
        LIMIT ?
        """,
            [limit],
        )

        for record in cur:
            data = record[0]
            obj = {} if not data else json.loads(data)
            keys = keys | extract_keys(obj)

        return extract_if_length_one(sorted(keys))

    @SKLearnEvaluationLogger.log("SQLiteTracker")
    def get_sample_query(self, compatibility_mode=True):
        keys = self.get_parameters_keys()

        if compatibility_mode or not ARROW_OPERATOR_SUPPORTED:
            TEMPLATE = TEMPLATE_JSON_EXTRACT
        else:
            TEMPLATE = TEMPLATE_ARROW

        template = Template(TEMPLATE)
        return template.render(keys=collapse(keys))

    @SKLearnEvaluationLogger.log("SQLiteTracker")
    def get(self, uuid, unserialize_plots=True):
        """Get an experiment given its UUID

        Examples
        --------
        >>> from sklearn_evaluation import SQLiteTracker
        >>> tracker = SQLiteTracker("experiments.db")
        >>> experiment = tracker.new_experiment() # new experiment
        >>> experiment.log("accuracy", 0.8) # log metric
        0.8
        >>> experiment = tracker.get(experiment.uuid) # retrieve it with the uuid
        >>> experiment
        Experiment({'accuracy': 0.8})
        >>> experiment.comment("best model")
        """
        cur = self.conn.execute(
            """
        SELECT parameters
        FROM experiments
        WHERE uuid = ?
        """,
            [uuid],
        )

        rows = cur.fetchone()

        if rows:
            obj = {} if not rows[0] else json.loads(rows[0])
            data = (
                obj
                if not unserialize_plots
                else {
                    k: unserialize_if_plot(v, return_instance=True)
                    for k, v in obj.items()
                }
            )
            return Experiment(tracker=self, uuid=uuid, data=data)
        else:
            raise ValueError("No record with such uuid")

    def __repr__(self):
        return self._recent(fmt="plain")

    def _repr_html_(self):
        return self._recent(fmt="html")

    def __del__(self):
        self.conn.close()

    def __len__(self):
        cur = self.conn.execute(
            """
        SELECT COUNT(*)
        FROM experiments
        """
        )
        return cur.fetchone()[0]


def is_str(obj):
    return isinstance(obj, str)


def is_float(obj):
    return isinstance(obj, float)


def json_loads(obj):
    try:
        value = json.loads(obj)
    except Exception:
        return False

    # strings like "123" are valid json objects but not in our context
    if not isinstance(value, (list, dict)):
        return False
    else:
        return value


def is_plot(obj):
    is_str_ = is_str(obj)

    if is_str_ or isinstance(obj, dict):
        if is_str_:
            obj_ = json_loads(obj)
        else:
            obj_ = obj

        if obj_:
            return "class" in obj_ and "version" in obj_
        else:
            return False
    else:
        return False


def unserialize_plot(obj, return_instance=False):
    if is_str(obj):
        obj = json.loads(obj)

    class_name = obj.pop("class", None)
    obj.pop("version", None)

    mod, _, attribute = class_name.rpartition(".")
    class_ = getattr(importlib.import_module(mod), attribute)
    instance = class_._from_data(**obj)

    return instance if return_instance else instance._repr_html_()


class GenericPlot:
    def __init__(self, html):
        self._html = html

    def _repr_html_(self):
        return self._html


def unserialize_if_plot(obj, return_instance=False):
    if is_plot(obj):
        return unserialize_plot(obj, return_instance=return_instance)
    elif is_str(obj) and obj.startswith("<img src="):
        return GenericPlot(obj)
    else:
        return obj


class Results:
    """An object to generate an HTML table from a SQLite result"""

    def __init__(self, columns, rows, render_plots):
        self.columns = columns
        self.rows = rows
        self.render_plots = render_plots

    def _repr_html_(self):
        return Template(
            """
<div>
<table>
  <thead>
  <tr>{% for name in columns %}<th>{{name}}</th>{% endfor %}</tr>
  </thead>
  <tbody>
  {% for row in rows %}
  <tr>
    {%- for field in row -%}
        {%- if is_str(field) and "<img src=" in field -%}
        <td>{{ "[Plot]" if not render_plots else field }}</td>
        {%- elif is_float(field) -%}
        <td>{{ "%.6f"| format(field) }}</td>
        {%- elif is_plot(field) -%}
        <td>{{ "[Plot]" if not render_plots else unserialize_plot(field) }}</td>
        {%- else -%}
        <td>{{field}}</td>
        {%- endif -%}
    {%- endfor -%}
  </tr>
  {% endfor %}
  </tbody>
</table>
</div>
"""
        ).render(
            columns=self.columns,
            rows=self.rows,
            render_plots=self.render_plots,
            is_str=is_str,
            is_float=is_float,
            is_plot=is_plot,
            unserialize_plot=unserialize_plot,
        )

    def __getitem__(self, key):
        if key not in self.columns:
            raise KeyError(f"{key} does not appear in the results")

        idx = self.columns.index(key)
        rows = [[row[idx]] for row in self.rows]

        return Results(columns=[key], rows=rows, render_plots=self.render_plots)

    def get(self, key, index_by=None):
        """Get a single column of the results

        Examples
        --------
        >>> from sklearn_evaluation import SQLiteTracker
        >>> tracker = SQLiteTracker("experiments.db")
        >>> exp1 = tracker.new_experiment()
        >>> exp1.log("accuracy", 0.8) # doctest: +SKIP
        >>> exp1.log_confusion_matrix([1, 1, 0, 0], [1, 0, 1, 0]) # doctest: +SKIP
        >>> exp2 = tracker.new_experiment()
        >>> exp2.log("accuracy", 1.0) # doctest: +SKIP
        >>> exp2.log_confusion_matrix([1, 1, 0, 0], [1, 1, 0, 0]) # doctest: +SKIP
        >>> results = tracker.query('''
        ... SELECT uuid,
        ...        json_extract(parameters, '$.accuracy') AS accuracy,
        ...        json_extract(parameters, '$.confusion_matrix') AS confusion_matrix
        ... FROM experiments
        ... ''', as_frame=False, render_plots=False)
        >>> results.get("confusion_matrix") # doctest: +SKIP
        """
        if key not in self.columns:
            raise KeyError(f"{key} does not appear in the results")

        if index_by is not None and index_by not in self.columns:
            raise KeyError(f"{index_by} does not appear in the results")

        if index_by is None:
            idx_id = 0
        else:
            idx_id = self.columns.index(index_by)

        idx = self.columns.index(key)
        values = [row[idx] for row in self.rows]
        ids = [format_id(row[idx_id]) for row in self.rows]

        e, ids_out = add_compare_tab(elements=values, ids=ids, scores_arg=None)
        e = [unserialize_if_plot(item, return_instance=False) for item in e]
        mapping = {k: v for k, v in zip(ids_out, e)}
        html = tabs_html_from_content(ids_out, e)

        return HTMLMapping(mapping, html)


def format_id(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, float):
        return f"{value:.6f}"
    else:
        return value


def extract_keys(d):
    return set(tuple(keys) for keys in _extract_keys(d))


def _extract_keys(d):
    """ """
    keys = []

    for key, value in d.items():
        if isinstance(value, Mapping):
            keys.extend([key] + tail for tail in _extract_keys(value))
        else:
            keys.append([key])

    return keys


def extract_if_length_one(elements):
    out = []

    for e in elements:
        if len(e) == 1:
            out.append(e[0])
        else:
            out.append(e)

    return out


def collapse(elements):
    out = []

    for e in elements:
        if isinstance(e, str):
            out.append((e, e))
        else:
            out.append((".".join(e), e[-1]))

    return out
