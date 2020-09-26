from uuid import uuid4
import sqlite3
import json

import pandas as pd
from sklearn_evaluation.table import Table


class SQLiteTracker:
    """A simple experiment tracker using SQLite

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

    def query(self, code):
        """Query the database, returns a pandas.DataFrame

        Examples
        --------
        >>> from sklearn_evaluation import SQLiteTracker
        >>> tracker = SQLiteTracker(':memory:') # example in-memory db
        >>> tracker.insert('my_uuid', {'a': 1})
        >>> df = tracker.query(
        ... "SELECT uuid, json_extract(parameters, '$.a') FROM experiments")
        """
        df = pd.read_sql(code, self.conn)
        if 'uuid' in df:
            df = df.set_index('uuid')
        return df

    def new(self):
        """Create a new experiment, returns a uuid
        """
        uuid = uuid4().hex
        cur = self.conn.cursor()
        cur.execute(
            """
        INSERT INTO experiments (uuid)
        VALUES(?)
        """, [uuid])
        cur.close()
        self.conn.commit()
        return uuid

    def update(self, uuid, parameters):
        """Update the parameters of an empty experiment given its uuid
        """
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

    def insert(self, uuid, parameters):
        """Insert a new experiment
        """
        cur = self.conn.cursor()
        cur.execute(
            """
        INSERT INTO experiments (uuid, parameters)
        VALUES(?, ?)
        """, [uuid, json.dumps(parameters)])
        cur.close()
        self.conn.commit()

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

    def __repr__(self):
        return self._recent(fmt='plain')

    def _repr_html_(self):
        return self._recent(fmt='html')

    def __del__(self):
        self.conn.close()
