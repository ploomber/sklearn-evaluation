from jinja2 import Template
from pathlib import Path

try:
    import mistune
except:
    raise ImportError('You need to install mistune to generate reports')


# import mistune
from sklearn_evaluation.report.util import jinja_env


class Report:

    def __init__(self, evaluator, template=None):
        self.evaluator = evaluator

        if template is None:
            template = jinja_env().get_template(self.evaluator.TEMPLATE_NAME)
        elif isinstance(template, Path):
            template = Template(template.read_text())
        elif isinstance(template, str):
            template = Template(template)

        rendered = template.render(e=evaluator)
        md = mistune.Markdown()
        self.rendered = md(rendered)

    def _repr_html_(self):
        return self.rendered

    def save(self, path):
        Path(path).write_text(self.rendered)
