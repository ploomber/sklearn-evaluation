from ..report.serialize import figure2html


class Plot:

    def _repr_html_(self):
        return figure2html(self.figure)
