import pytest
from sklearn_evaluation.report import ModelHeuristics, ReportSection
from sklearn_evaluation import plot
from sklearn_evaluation.report.util import validate_args_not_none

fpr = [0.0, 0.2, 0.4, 0.4, 0.6, 0.8, 1.0]
tpr = [0.0, 0.2, 0.4, 0.8, 0.8, 1.0, 1.0]


@pytest.mark.parametrize(
    "report_title, section_title, guidelines, plots, expected_html_elements",
    [
        [
            "template example",
            "test section",
            ["some text line 1", "more text line 2"],
            [],
            [
                "test section",
                "<h1>template example</h1>",
                "<li>some text line 1</li>",
                "<li>more text line 2</li>",
            ],
        ],
        [
            "template example with plots",
            "test section",
            ["some text line 1", "more text line 2"],
            [plot.ROC(tpr, fpr).plot().ax, plot.ROC(tpr, fpr).plot().ax],
            [
                "test section",
                "<h1>template example with plots</h1>",
                "<li>some text line 1</li>",
                "<li>more text line 2</li>",
            ],
        ],
    ],
)
def test_model_create_report_template(
    report_title, section_title, guidelines, plots, expected_html_elements
):
    mh = ModelHeuristics()
    test_section = ReportSection(section_title)

    for guideline in guidelines:
        test_section.append_guideline(guideline)
    for p in plots:
        test_section.append_guideline(p)

    test_section.set_include_in_report(True)
    mh._add_section_to_report(test_section)

    r = mh.create_report(title=report_title)
    report_html = r._repr_html_()

    assert report_title in report_html
    assert section_title in section_title

    for expected_html in expected_html_elements:
        assert expected_html in report_html

    assert report_html.count("<img src=") == len(plots)


@pytest.mark.parametrize(
    "guidelines, plots",
    [
        [
            ["some text line 1", "more text line 2"],
            [plot.ROC(tpr, fpr).plot().ax, plot.ROC(tpr, fpr).plot().ax],
        ],
        [[], [plot.ROC(tpr, fpr).plot().ax]],
        [["some text line 1", "more text line 2"], []],
    ],
)
def test_report_section_model(guidelines, plots):
    mh = ModelHeuristics()
    key = "test_section"

    test_section = ReportSection(key)

    for guideline in guidelines:
        test_section.append_guideline(guideline)

    for p in plots:
        test_section.append_guideline(p)

    test_section.set_include_in_report(True)
    mh._add_section_to_report(test_section)

    guidelines = mh.evaluation_state[key]["guidelines"]

    for i in range(len(guidelines)):
        guideline = guidelines[i]
        assert guideline == guidelines[i]

    for i in range(len(plots)):
        plot = plots[i]
        assert plot.__class__.__name__ == "Axes"


@pytest.mark.parametrize(
    "a, b, expected",
    [[1, None, None], [1, 1, True], [None, None, None], [None, 1, None]],
)
def test_validate_args_not_none(a, b, expected):
    @validate_args_not_none
    def func(a, b):
        return True

    assert func(a, b) is expected
