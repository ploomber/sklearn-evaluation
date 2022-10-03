from jinja2 import Template
from pathlib import Path
from sklearn_evaluation.telemetry import SKLearnEvaluationLogger

try:
    import mistune
except ModuleNotFoundError:
    mistune = None

from sklearn_evaluation.report.util import jinja_env


class Report:

    def __init__(self, evaluator, template=None):
        if mistune is None:
            raise ModuleNotFoundError(
                'You need to install mistune to generate reports')

        self.evaluator = evaluator

        if template is None:
            template = jinja_env().get_template(self.evaluator.TEMPLATE_NAME)
        elif isinstance(template, Path):
            template = Template(template.read_text())
        elif isinstance(template, str):
            template = Template(template)

        rendered = template.render(e=evaluator)

        if mistune.__version__[0] == '2':
            md = mistune.create_markdown()
        else:
            md = mistune.Markdown()

        self.rendered = md(rendered)

    def _repr_html_(self):
        return self.rendered

    @SKLearnEvaluationLogger.log(feature='report', action='save')
    def save(self, path):
        Path(path).write_text(self.rendered)
