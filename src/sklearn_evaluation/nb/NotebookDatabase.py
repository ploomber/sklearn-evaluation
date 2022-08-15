import json
import sqlite3
from glob import iglob

from .NotebookIntrospector import NotebookIntrospector


class NotebookDatabase:
    """
    Construct a SQLite database with a folder of notebooks
    """

    def __init__(self, path_to_db, pattern):
        self._path_to_db = path_to_db
        self._pattern = pattern

        self._conn = sqlite3.connect(self._path_to_db)

        cur = self._conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS nbs (
            created TIMESTAMP default current_timestamp,
            path TEXT NOT NULL PRIMARY KEY,
            c TEXT
        )
        """)
        cur.close()

    def index(self, verbose=True, update=False):
        """Index notebooks

        Parameters
        ----------
        verbose : bool, default=True
            If True, it prints one message per notebook

        update : bool, default=False
            If True, it updates the entry in the database if the path already
            exists
        """
        for path in iglob(self._pattern, recursive=True):
            try:
                nb = NotebookIntrospector(path)
            except Exception as e:
                if verbose:
                    print(f'Error loading {path}: {e}')
                continue

            paths_indexed = self._get_paths()

            if path not in paths_indexed:
                if verbose:
                    print(f'Indexing {path}')

                data = nb.to_json_serializable()

                cur = self._conn.cursor()
                cur.execute(
                    """
        INSERT INTO nbs (c, path)
        VALUES(?, ?)
        """, [
                        json.dumps(data),
                        path,
                    ])

                cur.close()

                self._conn.commit()
            elif path in paths_indexed and update:

                if verbose:
                    print(f'Updating {path}')

                data = nb.to_json_serializable()

                cur = self._conn.cursor()
                cur.execute(
                    """
        UPDATE nbs SET c = ?
        WHERE path = ?
        """, [
                        json.dumps(data),
                        path,
                    ])

                cur.close()

                self._conn.commit()
            elif verbose:
                print(f'{path} already indexed. Skipping...')

    def query(self, query):
        import pandas as pd
        return pd.read_sql(query, con=self._conn)

    def _get_paths(self):
        """Returns a set with all paths to the indexed notebooks
        """
        cur = self._conn.cursor()
        res = cur.execute('SELECT path FROM nbs')
        paths = set(row[0] for row in res.fetchall())
        cur.close()
        return paths