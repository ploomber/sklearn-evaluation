from uuid import uuid4
import sqlite3
import json

import pandas as pd


class SQLiteTracker:
    """
    A simple experiment tracker using SQLite

    """
    def __init__(self, database):
        self.conn = sqlite3.connect(database)

        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            uuid TEXT NOT NULL UNIQUE,
            created TIMESTAMP default current_timestamp,
            content TEXT,
            comment TEXT
        )
        """)
        cur.close()

    def get(self, uuid):
        """Get experiment with a given uuid
        """
        # TODO: use pandas to convert the JSON dictionary to columns
        # TODO: make it work for a list of uuids
        return pd.read_sql('SELECT * FROM experiments WHERE uuid = ?',
                           self.conn,
                           params=[uuid],
                           index_col='uuid')

    def query(self, code):
        """Query the database, returns a pandas.DataFrame
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
        return uuid

    def update(self, uuid, content):
        """Update the content of an empty experiment given its uuid
        """
        self._can_update(uuid)

        cur = self.conn.cursor()
        cur.execute(
            """
        UPDATE experiments
        SET content = ?
        WHERE uuid = ?
        """, [json.dumps(content), uuid])
        cur.close()

    def insert(self, uuid, content):
        """Insert a new experiment
        """
        cur = self.conn.cursor()
        cur.execute(
            """
        INSERT INTO experiments (uuid, content)
        VALUES(?, ?)
        """, [uuid, json.dumps(content)])
        cur.close()

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

    def _can_update(self, uuid):
        """Check if an experiment with a given uuid can be updated
        """
        cur = self.conn.cursor()

        cur.execute(
            """
        SELECT content
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

    # TODO: implement a function to get the latest experiments
    # TODO: implement a useful __repr__ (maybe show most recent experiments?)
    # TODO: implement _repr_html_
