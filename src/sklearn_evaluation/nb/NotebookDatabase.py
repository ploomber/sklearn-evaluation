import json
import sqlite3
from glob import iglob
import nbformat

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
            -- uuid TEXT NOT NULL UNIQUE,
            created TIMESTAMP default current_timestamp,
            path TEXT NOT NULL,
            c TEXT
        )
        """)
        cur.close()

    def index(self):
        """Index notebooks
        """
        for path in iglob(self._pattern):
            try:
                nb = NotebookIntrospector(path)
            except Exception as e:
                print(f'Error loading {path}: {e}')
                continue

            metadata = nb.nb.metadata.get('sklearn-evaluation',
                                          dict(indexed=False))

            if not metadata['indexed']:
                print(f'Indexing {path}')
                nb.nb.metadata['sklearn-evaluation'] = dict(indexed=True)
                nbformat.write(nb.nb, path)

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
            else:
                print(f'{path} already indexed. Skipping...')

    def query(self, query):
        import pandas as pd
        return pd.read_sql(query, con=self._conn)
