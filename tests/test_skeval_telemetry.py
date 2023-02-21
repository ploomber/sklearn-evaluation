import pytest
import numpy as np
from unittest.mock import Mock, call

from ploomber_core.telemetry import telemetry
from sklearn_evaluation.telemetry import SKLearnEvaluationLogger
from sklearn_evaluation.plot.pca import pca
from sklearn_evaluation.plot.calibration import CalibrationCurve

MOCK_API_KEY = "phc_P1dsjk20bijsabdaib2eu"


@pytest.fixture
def mock_telemetry(monkeypatch):
    telemetry.Telemetry("ploomber", "0.14.0", MOCK_API_KEY)

    mock = Mock()
    mock_dt = Mock()
    mock_dt.now.side_effect = [1, 2]

    monkeypatch.setattr(telemetry.Telemetry, "log_api", mock)
    monkeypatch.setattr(telemetry.datetime, "datetime", mock_dt)
    monkeypatch.setattr(telemetry.sys, "argv", ["/path/to/bin", "arg"])

    yield mock


@pytest.mark.parametrize("action", ["some_action", None])
@pytest.mark.parametrize("feature", ["report", None])
@pytest.mark.parametrize("x", [1, None])
@pytest.mark.parametrize("y", [2, None])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"is_report": True, "not-a-flag": True},
        {"not-a-flag": "random-value"},
        {"is_report": True},
        {},
    ],
)
def test_logger(mock_telemetry, action, feature, x, y, kwargs):
    flags = dict()
    function_arguments = dict({"x": x, "y": y})
    _action = action or "my_function"

    @SKLearnEvaluationLogger.log(feature=feature, action=action)
    def my_function(a, b, x=None, y=None, **kwargs):
        pass

    my_function(1, 2, x=x, y=y, **kwargs)

    if len(kwargs) > 0:
        for key, value in kwargs.items():
            if key in SKLearnEvaluationLogger.flags():
                flags[key] = value

    expected_metadata = dict(
        {"action": _action, "feature": feature, "args": function_arguments}
    )

    if len(flags) > 0:
        expected_metadata["flags"] = flags

    mock_telemetry.assert_has_calls(
        [call("sklearn-evaluation", metadata=expected_metadata)]
    )


@pytest.mark.parametrize("action", ["some_action", None])
@pytest.mark.parametrize("feature", ["report", None])
@pytest.mark.parametrize("x", [1, None])
@pytest.mark.parametrize("y", [2, None])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"is_report": True, "not-a-flag": True},
        {"not-a-flag": "random-value"},
        {"is_report": True},
        {},
    ],
)
def test_logger_with_errors(mock_telemetry, action, feature, x, y, kwargs):
    _exception = "Failed to run function"
    flags = dict()
    function_arguments = dict({"x": x, "y": y})
    _action = action or "my_function"

    @SKLearnEvaluationLogger.log(feature=feature, action=action)
    def my_function(a, b, x=None, y=None, **kwargs):
        raise Exception(_exception)

    try:
        my_function(1, 2, x=x, y=y, **kwargs)
    except Exception:
        pass

    if len(kwargs) > 0:
        for key, value in kwargs.items():
            if key in SKLearnEvaluationLogger.flags():
                flags[key] = value

    expected_metadata = dict(
        {
            "action": _action,
            "feature": feature,
            "args": function_arguments,
            "exception": _exception,
        }
    )

    if len(flags) > 0:
        expected_metadata["flags"] = flags

    mock_telemetry.assert_has_calls(
        [call("sklearn-evaluation-error", metadata=expected_metadata)]
    )


def test_pca(mock_telemetry):

    pca(np.array([[1, 3, 2], [3, 0, 6]]))
    function_arguments = dict(
        {"y": None, "target_names": None, "n_components": 2, "colors": None, "ax": None}
    )
    expected_metadata = dict(
        {"action": "pca", "feature": "plot", "args": function_arguments}
    )

    mock_telemetry.assert_has_calls(
        [call("sklearn-evaluation", metadata=expected_metadata)]
    )


def test_calibration_curve(mock_telemetry):

    CalibrationCurve([0.5, 0.9], [0.45, 0.89], label=["Classifier 1"]).plot()
    function_arguments = dict({"label": ["Classifier 1"], "cmap": None})
    expected_metadata = dict(
        {
            "action": "calibration-curve-init",
            "feature": "plot",
            "args": function_arguments,
        }
    )
    mock_telemetry.assert_has_calls(
        [call("sklearn-evaluation", metadata=expected_metadata)]
    )


def test_calibration_curve_add(mock_telemetry):

    cc1 = CalibrationCurve([0.5, 0.9], [0.45, 0.89], label=["Classifier 1"]).plot()
    cc2 = CalibrationCurve([0.49, 0.92], [0.52, 0.88], label=["Classifier 2"]).plot()
    cc1 + cc2
    function_arguments_one = dict({"label": ["Classifier 1"], "cmap": None})
    expected_metadata_one = dict(
        {
            "action": "calibration-curve-init",
            "feature": "plot",
            "args": function_arguments_one,
        }
    )

    function_arguments_two = dict({"label": ["Classifier 2"], "cmap": None})
    expected_metadata_two = dict(
        {
            "action": "calibration-curve-init",
            "feature": "plot",
            "args": function_arguments_two,
        }
    )

    function_arguments_add = dict({"cmaps": [None, None]})
    expected_metadata_add = dict(
        {
            "action": "calibration-curve-add-init",
            "feature": "plot",
            "args": function_arguments_add,
        }
    )

    mock_telemetry.assert_has_calls(
        [
            call("sklearn-evaluation", metadata=expected_metadata_one),
            call("sklearn-evaluation", metadata=expected_metadata_two),
            call("sklearn-evaluation", metadata=expected_metadata_add),
        ]
    )
