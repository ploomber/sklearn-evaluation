from uuid import uuid4
import sqlite3
import json

import pandas as pd
from jinja2 import Template

from sklearn_evaluation.table import Table
from sklearn_evaluation.report.serialize import try_serialize_figures
from sklearn_evaluation.telemetry import SKLearnEvaluationLogger
from sklearn_evaluation.nb.NotebookCollection import (add_compare_tab,
                                                      tabs_html_from_content,
                                                      HTMLMapping)

minor = sqlite3.sqlite_version.split('.')[1]

# -> and ->> operators introduced in sqlite version 3.38.0
# https://www.sqlite.org/json1.html#jptr
ARROW_OPERATOR_SUPPORTED = int(minor) >= 38

TEMPLATE_ARROW = """\
SELECT
    uuid,
    {% for k in keys -%}
    parameters ->> '{{k}}' as {{k}}{% if not loop.last %},{% endif %}
    {% endfor -%}
FROM experiments
LIMIT 10
"""

TEMPLATE_JSON_EXTRACT = """\
SELECT
    uuid,
    {% for k in keys -%}
    json_extract(parameters, '$.{{k}}') as {{k}}{% if not loop.last %},{% endif %}
    {% endfor -%}
FROM experiments
LIMIT 10
"""


class SQLiteTracker:
    """A simple experiment tracker using SQLite

    :doc:`Click here <../user_guide/SQLiteTracker>` to see the user guide.

    Parameters
    ----------
    path
        Database location

    """

    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)

        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            uuid TEXT NOT NULL UNIQUE,
            created TIMESTAMP default current_timestamp,
            parameters TEXT,
            comment TEXT
        )
        """)
        cur.close()

    def __getitem__(self, uuid):
        """Get experiment with a given uuid
        """
        # TODO: make it work for a list of uuids
        return pd.read_sql('SELECT * FROM experiments WHERE uuid = ?',
                           self.conn,
                           params=[uuid],
                           index_col='uuid')

    def recent(self, n=5, normalize=False):
        """Get most recent experiments as a pandas.DataFrame
        """
        query = """
        SELECT uuid, created, parameters, comment
        FROM experiments
        ORDER BY created DESC
        LIMIT ?
        """
        df = pd.read_sql(query, self.conn, params=[n], index_col='uuid')

        if normalize:
            # parse and normalize json
            parameters = pd.json_normalize(
                df.pop('parameters').apply(lambda s: json.loads(s))).set_index(
                    df.index)
            df = df.join(parameters)

            # re order columns to show "comment" at the end
            comment = df.pop('comment')
            df.insert(len(df.columns), 'comment', comment)

        return df

    @SKLearnEvaluationLogger.log(feature='SQLiteTracker')
    def query(self, code, as_frame=True, render_plots=False):
        """Query the database, returns a pandas.DataFrame

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
        >>> tracker = SQLiteTracker(':memory:') # example in-memory db
        >>> tracker.insert('my_uuid', {'a': 1})
        >>> df = tracker.query(
        ... "SELECT uuid, json_extract(parameters, '$.a') FROM experiments")
        """
        if as_frame:
            df = pd.read_sql(code, self.conn)

            if 'uuid' in df:
                df = df.set_index('uuid')

            return df
        else:
            cursor = self.conn.execute(code)
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            return Results(columns, rows, render_plots=render_plots)

    @SKLearnEvaluationLogger.log(feature='SQLiteTracker')
    def new(self):
        """Create a new experiment, returns a uuid
        """
        uuid = str(uuid4())[:8]
        cur = self.conn.cursor()
        cur.execute(
            """
        INSERT INTO experiments (uuid)
        VALUES(?)
        """, [uuid])
        cur.close()
        self.conn.commit()
        return uuid

    @SKLearnEvaluationLogger.log(feature='SQLiteTracker')
    def update(self, uuid, parameters, allow_overwrite=False):
        """Update the parameters of a experiment given its uuid
        """
        if not allow_overwrite:
            self._can_update(uuid)

        cur = self.conn.cursor()
        cur.execute(
            """
        UPDATE experiments
        SET parameters = ?
        WHERE uuid = ?
        """, [json.dumps(parameters), uuid])
        cur.close()
        self.conn.commit()

    def upsert(self, uuid, parameters):
        existing = self._get(uuid)

        cur = self.conn.cursor()
        cur.execute(
            """
        UPDATE experiments
        SET parameters = ?
        WHERE uuid = ?
        """, [json.dumps({
                **existing,
                **parameters
            }), uuid])
        cur.close()
        self.conn.commit()

    @SKLearnEvaluationLogger.log('SQLiteTracker')
    def insert(self, uuid, parameters):
        """Insert a new experiment
        """
        # serialize matplotlib.figure.Figure, if any
        parameters = try_serialize_figures(parameters)

        cur = self.conn.cursor()
        cur.execute(
            """
        INSERT INTO experiments (uuid, parameters)
        VALUES(?, ?)
        """, [uuid, json.dumps(parameters)])
        cur.close()
        self.conn.commit()

    @SKLearnEvaluationLogger.log('SQLiteTracker')
    def comment(self, uuid, comment):
        """Add a comment to an experiment given its uuid
        """
        # TODO: add overwrite (false by default) and append options
        cur = self.conn.cursor()
        cur.execute(
            """
        UPDATE experiments
        SET comment = ?
        WHERE uuid = ?
        """, [comment, uuid])
        cur.close()
        self.conn.commit()

    def _recent(self, n=5, fmt='html'):
        if fmt not in {'html', 'plain'}:
            raise ValueError('fmt must be one "html" or "plain"')

        cur = self.conn.cursor()
        cur.execute(
            """
        SELECT uuid, created, parameters, comment
        FROM experiments
        ORDER BY created DESC
        LIMIT ?
        """, [n])
        res = cur.fetchall()
        table = Table(res, header=['uuid', 'created', 'parameters', 'comment'])

        title_template = '<h4> {} </h4>' if fmt == 'html' else '{}\n'
        title = title_template.format(type(self).__name__)

        if not len(table):
            title += '(No experiments saved yet)'
            if fmt == 'plain':
                title += '\n'

        if len(table):
            footer = (('<br>' if fmt == 'html' else '\n') +
                      '(Most recent experiments)')
        else:
            footer = ''

        return (title + (table.to_html() if fmt == 'html' else str(table)) +
                footer)

    def _can_update(self, uuid):
        """Check if an experiment with a given uuid can be updated
        """
        cur = self.conn.cursor()

        cur.execute(
            """
        SELECT parameters
        FROM experiments
        WHERE uuid = ?
        """, [uuid])

        row = cur.fetchone()

        exists = row is not None

        if exists:
            empty = row[0] is None

            if not empty:
                raise ValueError('Cannot update non-empty experiment with '
                                 'uuid "{}"'.format(uuid))
        else:
            raise ValueError('Cannot update experiment with '
                             'uuid "{}" because it does '
                             'not exist'.format(uuid))

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
        """, [limit])

        for record in cur:
            obj = json.loads(record[0])
            keys = keys | set(obj)

        return sorted(keys)

    def get_sample_query(self, compatibility_mode=True):
        keys = self.get_parameters_keys()

        if compatibility_mode or not ARROW_OPERATOR_SUPPORTED:
            TEMPLATE = TEMPLATE_JSON_EXTRACT
        else:
            TEMPLATE = TEMPLATE_ARROW

        template = Template(TEMPLATE)
        return template.render(keys=keys)

    def _get(self, uuid):
        cur = self.conn.execute(
            """
        SELECT parameters
        FROM experiments
        WHERE uuid = ?
        """, [uuid])

        rows = cur.fetchone()

        if rows:
            return json.loads(rows[0])
        else:
            raise ValueError("No record with such uuid")

    def __repr__(self):
        return self._recent(fmt='plain')

    def _repr_html_(self):
        return self._recent(fmt='html')

    def __del__(self):
        self.conn.close()


def is_str(obj):
    return isinstance(obj, str)


def is_float(obj):
    return isinstance(obj, float)


class Results:
    """An object to generate an HTML table from a SQLite result
    """

    def __init__(self, columns, rows, render_plots):
        self.columns = columns
        self.rows = rows
        self.render_plots = render_plots

    def _repr_html_(self):
        return Template("""
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
        {%- else -%}
        <td>{{field}}</td>
        {%- endif -%}
    {%- endfor -%}
  </tr>
  {% endfor %}
  </tbody>
</table>
</div>
""").render(columns=self.columns,
            rows=self.rows,
            render_plots=self.render_plots,
            is_str=is_str,
            is_float=is_float)

    def __getitem__(self, key):
        if key not in self.columns:
            raise KeyError(f'{key} does not appear in the results')

        idx = self.columns.index(key)
        rows = [[row[idx]] for row in self.rows]

        return Results(columns=[key],
                       rows=rows,
                       render_plots=self.render_plots)

    def get(self, key, index_by=None):
        if key not in self.columns:
            raise KeyError(f'{key} does not appear in the results')

        if index_by is not None and index_by not in self.columns:
            raise KeyError(f'{index_by} does not appear in the results')

        if index_by is None:
            idx_id = 0
        else:
            idx_id = self.columns.index(index_by)

        idx = self.columns.index(key)
        values = [row[idx] for row in self.rows]
        ids = [format_id(row[idx_id]) for row in self.rows]

        e, ids_out = add_compare_tab(elements=values, ids=ids, scores_arg=None)
        mapping = {k: v for k, v in zip(ids_out, e)}
        html = tabs_html_from_content(ids_out, e)
        return HTMLMapping(mapping, html)


def format_id(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, float):
        return f'{value:.6f}'
    else:
        return value
