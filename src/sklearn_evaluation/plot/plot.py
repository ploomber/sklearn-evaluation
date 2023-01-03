import json
from pathlib import Path


class Plot:
    # Deleted serializaiton function as proof of fix

    def dump(self, path):
        data = self._get_data()
        Path(path).write_text(json.dumps(data), encoding="utf-8")
