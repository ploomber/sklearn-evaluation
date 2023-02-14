from sklearn_evaluation.report.serialize import EvaluatorHTMLSerializer
from sklearn_evaluation.report.report import Report
from jinja2 import Template
from sklearn_evaluation.report.serialize import figure2html
import abc


class ReportSection:
    """
    Section to include in report
    """

    def __init__(self, key, include_in_report=True):
        self.report_section = dict(
            {
                "guidelines": [],
                "title": key.replace("_", " "),
                "include_in_report": include_in_report,
                "is_ok": False,
            }
        )
        self.key = key

    def append_guideline(self, guideline):
        """
        Add guideline to section

        Parameters
        ----------
        guideline : str
            The guideline to add
        """
        self.report_section["guidelines"].append(guideline)

    def get_dict(self) -> dict:
        """
        Return dict of the section
        """
        return self.report_section

    def set_is_ok(self, is_ok):
        """
        Set if the reported test is valid
        """
        self.report_section["is_ok"] = is_ok

    def set_include_in_report(self, include):
        """
        Set if should include this section in the report
        """
        self.report_section["include_in_report"] = include


class ModelHeuristics(abc.ABC):
    """
    Base class for generating model heuristics and reports
    """

    _report_css_style = """
    .model-evaluation-container h1 {
        font-size: 2.25em;
        margin-bottom: 0;
    }

    .model-evaluation-container h2 {
        font-size: 1.25em;
    }

    .model-evaluation-container {
        font-family: Helvetica, sans-serif, Arial;
        text-align: left;
        width: fit-content;
        margin: 50px auto;
    }

    .model-evaluation-container .block {
        margin-bottom: 0px;
        border-bottom: 1px solid #e5e4e4;
        padding: 0.75em 0;
    }

    .model-evaluation-container .nobull {
        list-style-type: none;
    }

    .model-evaluation-container ul li {
        margin-bottom: 10px;
    }

    .model-evaluation-container ul {
        padding: 0;
    }

    .model-evaluation-container ul li:not(.nobull) {
        margin-left: 1em;
    }

    .model-evaluation-container .display-inline-block {
        display: inline-block;
    }

    .model-evaluation-container .capitalize {
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

                {% for key in evaluation_state.keys() %}
                <div class="block">
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
                </div>
                {% endfor %}

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

    def _get_model_name(self, model) -> str:
        """
        Returns model name
        """
        return model.__class__.__name__
