from sklearn_evaluation.report.serialize import EvaluatorHTMLSerializer
from sklearn_evaluation.report.report import Report
from jinja2 import Template
from sklearn_evaluation.report.serialize import figure2html
import abc


class ModelHeuristics(abc.ABC):
    """
    Base class for generating model heuristics and reports
    """

    _report_css_style = """
    .model-evaluation-container {
        font-family: Arial, Helvetica, sans-serif;
        text-align: left;
        width: fit-content;
        margin: 50px auto;
    }

    .block {
        margin-bottom: 50px
    }

    .nobull {
        list-style-type: none;
    }

    ul li {
        margin-bottom: 10px;
    }

    .display-inline-block {
        display: inline-block;
    }

    .capitalize {
        text-transform: capitalize;
    }
    """

    def __init__(self):
        """
        Initialize ModelHeuristics
        """
        self.evaluation_state = dict()

    def _add_section_to_report(self, section):
        """
        Add section to report

        Parameters
        ----------
        section : ReportSection
            The section to include
        """
        self.evaluation_state[section.key] = section.get_dict()

    def _get_calculate_failed_error(self, key, model_name, exc=None) -> str:
        """
        Include the raised error in the report

        Parameters
        ----------
        key : str
            The action key

        model_name : str
            The model name

        exc : Expection, default = None
            Include the error message

        Returns
        -------
        formatted error to display in the report str
        """
        guideline = f"Failed to calculate {key} for {model_name}"
        if exc:
            exc_message = getattr(exc, "message", repr(exc))
            guideline += f"\nWith message : {exc_message}"

        return guideline

    def create_report(self, title) -> Report:
        """
        Generates html report

        Parameters
        ----------
        title : str
            Report title

        Returns
        -------
        Report instance
        """
        e = EvaluatorHTMLSerializer(None)
        template = Template(
            """
<html>
<head>
    <style>
    {{style}}
        </style>

    </head>
<body>
    <div class="model-evaluation-container">
        <div>
            <h1>{{title}}</h1>

            <div class="block">
                {% for key in evaluation_state.keys() %}
                    {% if evaluation_state[key]["include_in_report"] %}
                    <ul>
                        <li class="nobull"><h2 class="capitalize">
                        {{evaluation_state[key]["title"]}}</h2></li>
                        {% for guideline in evaluation_state[key]["guidelines"] %}

                            {% if guideline is string %}
                                <li>{{guideline}}</li>
                            {% else %}
                                <p class="display-inline-block">
                                {{figure2html(guideline.get_figure())}}</p>
                            {% endif %}
                        {% endfor %}
                    </ul>
                    {% endif %}

                {% endfor %}
            </div>

        </div>
    </div>
</body>
</html>
    """
        ).render(
            title=title,
            evaluation_state=self.evaluation_state,
            figure2html=figure2html,
            style=self._report_css_style,
        )

        report = Report(e, template)
        return report
