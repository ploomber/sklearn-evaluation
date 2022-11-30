import json
from pathlib import Path

from ..report.serialize import figure2html


class Plot:
    def _repr_html_(self):
        return figure2html(self.figure)

    def dump(self, path):
        data = self._get_data()
        Path(path).write_text(json.dumps(data), encoding="utf-8")
