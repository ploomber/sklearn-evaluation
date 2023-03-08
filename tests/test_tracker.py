import json
from uuid import uuid4
import time

import pytest
from sklearn_evaluation.tracker import SQLiteTracker
from sklearn_evaluation import tracker as tracker_module
from sklearn_evaluation import plot


def test_insert():
    tracker = SQLiteTracker(":memory:")
    tracker.insert("some_uuid", {"a": 1})

    res = tracker.query('SELECT * FROM experiments WHERE uuid = "some_uuid"')

    assert len(res) == 1


def test_new():
    tracker = SQLiteTracker(":memory:")
    assert isinstance(tracker.new(), str)


def test_comment():
    tracker = SQLiteTracker(":memory:")
    uuid = tracker.new()
    tracker.comment(uuid, "this is a comment")
    res = tracker[uuid]

    assert res.loc[uuid].comment == "this is a comment"


def test_getitem():
    tracker = SQLiteTracker(":memory:")
    tracker.insert("some_uuid", {"a": 1})
    res = tracker["some_uuid"]
    parameters = res.loc["some_uuid"].parameters

    assert len(res) == 1
    assert parameters == '{"a": 1}'


def test_update_errors():
    tracker = SQLiteTracker(":memory:")

    with pytest.raises(ValueError) as excinfo:
        tracker.update("unknown_uuid", {})

    assert "not exist" in str(excinfo.value)

    tracker.insert("some_uuid", {})

    with pytest.raises(ValueError) as excinfo:
        tracker.update("some_uuid", {})

    assert "non-empty" in str(excinfo.value)


def test_update():
    tracker = SQLiteTracker(":memory:")
    uuid = tracker.new()
    tracker.update(uuid, {})
    res = tracker[uuid]
    assert len(res) == 1


def test_update_override():
    tracker = SQLiteTracker(":memory:")
    tracker.insert("uuid", dict(a=0, b=0))
    tracker.update("uuid", dict(a=1, b=2), allow_overwrite=True)
    res = tracker.get("uuid")
    assert res == dict(a=1, b=2)


def test_get():
    tracker = SQLiteTracker(":memory:")
    uuid = tracker.new()
    tracker.update(uuid, dict(a=1, b=2))
    assert tracker.get(uuid) == dict(a=1, b=2)


def test_get_error():
    tracker = SQLiteTracker(":memory:")

    with pytest.raises(ValueError):
        tracker.get("uuid")


def test_upsert():
    tracker = SQLiteTracker(":memory:")

    uuid = tracker.new()
    tracker.update(uuid, dict(a=1, b=2))
    tracker.upsert(uuid, dict(a=2, c=3))

    assert tracker.get(uuid) == dict(a=2, b=2, c=3)
    assert len(tracker) == 1


def test_upsert_append():
    tracker = SQLiteTracker(":memory:")

    uuid = tracker.new()
    tracker.update(uuid, dict(a=1, b=2, c=[1], d=[4]))
    tracker.upsert_append(uuid, dict(a=2, c=3, d=[5], e=[6]))

    assert tracker.get(uuid) == dict(a=[1, 2], b=2, c=[1, 3], d=[4, 5], e=[6])
    assert len(tracker) == 1


def test_reprs():
    tracker = SQLiteTracker(":memory:")
    assert "SQLiteTracker" in repr(tracker)
    assert "SQLiteTracker" in tracker._repr_html_()
    assert "(No experiments saved yet)" in repr(tracker)
    assert "(No experiments saved yet)" in tracker._repr_html_()

    uuids = [uuid4().hex for _ in range(6)]

    for i, uuid in enumerate(uuids):
        tracker.insert(uuid, {"a": i})
        if i == 0:
            # create a delay so the timestamp for the 1st value
            # would be different from others
            time.sleep(3)

    expected = [False, True, True, True, True, True]

    assert [uuid in repr(tracker) for uuid in uuids] == expected
    assert [uuid in tracker._repr_html_() for uuid in uuids] == expected


def test_recent():
    tracker = SQLiteTracker(":memory:")

    for i in range(5):
        tracker.insert(i, {"a": i})

    df = tracker.recent(normalize=False)

    assert df.columns.tolist() == ["created", "parameters", "comment"]
    assert df.index.name == "uuid"

    df = tracker.recent(normalize=True)

    assert df.columns.tolist() == ["created", "a", "comment"]
    assert df.index.name == "uuid"


@pytest.mark.parametrize(
    "mapping, keys",
    [
        [dict(a=1, b=2), {("a",), ("b",)}],
        [dict(a=1, b=dict(c=2)), {("a",), ("b", "c")}],
        [dict(a=1, b=dict(c=dict(d=2))), {("a",), ("b", "c", "d")}],
    ],
)
def test_extract_keys(mapping, keys):
    assert tracker_module.extract_keys(mapping) == keys


def test_get_parameters_keys():
    tracker = SQLiteTracker(":memory:")

    to_insert = [
        dict(a=1, b=2),
        dict(x=1, y=2),
        dict(z=3),
    ]

    for i, data in enumerate(to_insert):
        tracker.insert(i, data)

    assert tracker.get_parameters_keys() == ["a", "b", "x", "y", "z"]


def test_get_parameters_keys_nested():
    tracker = SQLiteTracker(":memory:")

    to_insert = [
        dict(a=1, b=2),
        dict(a=dict(b=1, c=2), b=dict(c=2, d=3)),
        dict(a=dict(b=1, c=dict(d=2)), b=2),
    ]

    tracker.insert_many(to_insert)

    assert tracker.get_parameters_keys() == [
        "a",
        ("a", "b"),
        ("a", "c"),
        ("a", "c", "d"),
        "b",
        ("b", "c"),
        ("b", "d"),
    ]


expected_arrow = """\
SELECT
    uuid,
    parameters ->> 'a' as a,
    parameters ->> 'b' as b,
    parameters ->> 'x' as x,
    parameters ->> 'y' as y,
    parameters ->> 'z' as z
    FROM experiments
LIMIT 10\
"""

expected_json_extract = """\
SELECT
    uuid,
    json_extract(parameters, '$.a') as a,
    json_extract(parameters, '$.b') as b,
    json_extract(parameters, '$.x') as x,
    json_extract(parameters, '$.y') as y,
    json_extract(parameters, '$.z') as z
    FROM experiments
LIMIT 10\
"""

_to_insert = [
    dict(a=1, b=2),
    dict(x=1, y=2),
    dict(z=3),
]


expected_arrow_nested = """\
SELECT
    uuid,
    parameters ->> 'a.b.c' as c,
    parameters ->> 'x' as x,
    parameters ->> 'y' as y
    FROM experiments
LIMIT 10\
"""

expected_json_extract_nested = """\
SELECT
    uuid,
    json_extract(parameters, '$.a.b.c') as c,
    json_extract(parameters, '$.x') as x,
    json_extract(parameters, '$.y') as y
    FROM experiments
LIMIT 10\
"""

_to_insert_nested = [
    dict(a=dict(b=dict(c=1))),
    dict(x=1, y=2),
]


@pytest.mark.parametrize(
    "arrow_operator_supported, expected, to_insert",
    [
        [False, expected_json_extract, _to_insert],
        [True, expected_arrow, _to_insert],
        [False, expected_json_extract_nested, _to_insert_nested],
        [True, expected_arrow_nested, _to_insert_nested],
    ],
    ids=[
        "arrow-not-supported",
        "arrow-supported",
        "arrow-not-supported-nested",
        "arrow-supported-nested",
    ],
)
def test_get_sample_query(arrow_operator_supported, expected, to_insert, monkeypatch):
    monkeypatch.setattr(
        tracker_module, "ARROW_OPERATOR_SUPPORTED", arrow_operator_supported
    )

    tracker = SQLiteTracker(":memory:")
    tracker.insert_many(to_insert)

    assert tracker.get_sample_query(compatibility_mode=False) == expected


def test_new_experiment():
    tracker = SQLiteTracker(":memory:")
    assert len(tracker) == 0

    tracker.new_experiment()
    assert len(tracker) == 1

    tracker.new_experiment()
    assert len(tracker) == 2


def test_experiment_getitem():
    tracker = SQLiteTracker(":memory:")
    exp = tracker.new_experiment()
    exp.log("key", "value")

    retrieved = tracker.get(exp.uuid)
    assert retrieved["key"] == "value"


def test_experiment_log():
    tracker = SQLiteTracker(":memory:")
    experiment = tracker.new_experiment()
    experiment.log("accuracy", 0.8)

    retrieved = tracker.get(experiment.uuid)

    assert retrieved._data == {"accuracy": 0.8}
    assert len(tracker) == 1


def test_experiment_log_dict():
    tracker = SQLiteTracker(":memory:")

    experiment = tracker.new_experiment()
    experiment.log("accuracy", 0.8)
    experiment.log_dict(
        {"precision": 0.7, "recall": 0.6},
    )

    retrieved = tracker.get(experiment.uuid)

    assert retrieved._data == {"accuracy": 0.8, "precision": 0.7, "recall": 0.6}
    assert len(tracker) == 1


def test_experiment_log_figure():
    tracker = SQLiteTracker(":memory:")
    experiment = tracker.new_experiment()
    ax = plot.confusion_matrix([1, 1, 0, 0], [1, 0, 1, 0])
    experiment.log_figure("confusion_matrix", ax.figure)
    retrieved = tracker.get(experiment.uuid)

    # this will force unserialization
    assert tracker.query(
        """
SELECT  json_extract(parameters, '$.confusion_matrix') AS cm
FROM experiments
""",
        as_frame=False,
        render_plots=True,
    )._repr_html_()

    assert retrieved["confusion_matrix"]
    assert len(tracker) == 1


def test_experiment_log_confusion_matrix():
    tracker = SQLiteTracker(":memory:")
    experiment = tracker.new_experiment()
    experiment.log_confusion_matrix([1, 1, 0, 0], [1, 0, 1, 0])

    retrieved = tracker.get(experiment.uuid)

    # this will force unserialization
    assert tracker.query(
        """
SELECT  json_extract(parameters, '$.confusion_matrix') AS cm
FROM experiments
""",
        as_frame=False,
        render_plots=True,
    )._repr_html_()

    assert retrieved["confusion_matrix"]
    assert len(tracker) == 1


def test_experiment_log_classification_report():
    tracker = SQLiteTracker(":memory:")
    experiment = tracker.new_experiment()
    experiment.log_classification_report([1, 1, 0, 0], [1, 0, 1, 0])

    retrieved = tracker.get(experiment.uuid)

    # this will force unserialization
    assert tracker.query(
        """
SELECT  json_extract(parameters, '$.classification_report') AS cm
FROM experiments
""",
        as_frame=False,
        render_plots=True,
    )._repr_html_()

    assert retrieved["classification_report"]
    assert len(tracker) == 1


def test_experiment_comment():
    tracker = SQLiteTracker(":memory:")
    experiment = tracker.new_experiment()
    experiment.comment("some comment")

    df = tracker.query(
        """
SELECT uuid,
       comment
FROM experiments
WHERE comment IS NOT NULL
LIMIT 1
"""
    )

    assert df.to_dict() == {"comment": {experiment.uuid: "some comment"}}
    assert len(tracker) == 1


def test_html_render_with_numeric_looking_uuid():
    tracker = SQLiteTracker(":memory:")

    # edge case: uuid.uuid4() might generate all numbers
    tracker.insert("123", dict(a=1))

    results = tracker.query(
        """
SELECT uuid
FROM experiments
""",
        as_frame=False,
        render_plots=True,
    )

    results._repr_html_()


@pytest.mark.parametrize(
    "value, expected",
    [
        ["abcd", False],
        ["1234", False],
        [{"class": "something", "version": "something"}, True],
    ],
)
def test_is_plot(value, expected):
    assert tracker_module.is_plot(value) == expected


obj = {"class": "something", "version": "something"}


@pytest.mark.parametrize(
    "value, expected",
    [
        ["abcd", False],
        ["1234", False],
        [json.dumps(obj), obj],
    ],
)
def test_json_loads(value, expected):
    assert tracker_module.json_loads(value) == expected


def test_insert_many():
    tracker = SQLiteTracker(":memory:")

    experiments = [
        dict(a=1, b=2),
        dict(a=2, b=3),
        dict(a=3, b=4),
    ]

    assert not len(tracker)

    tracker.insert_many(experiments)

    assert len(tracker) == 3
