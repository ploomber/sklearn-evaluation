from uuid import uuid4
import time

import pytest
from sklearn_evaluation.tracker import SQLiteTracker
from sklearn_evaluation import tracker as tracker_module


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
    tracker.new()
    tracker.update(uuid, dict(a=1, b=2))
    tracker.upsert(uuid, dict(a=2, c=3))

    assert tracker.get(uuid) == dict(a=2, b=2, c=3)


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


def test_get_schema():
    tracker = SQLiteTracker(":memory:")

    to_insert = [
        dict(a=1, b=2),
        dict(x=1, y=2),
        dict(z=3),
    ]

    for i, data in enumerate(to_insert):
        tracker.insert(i, data)

    assert tracker.get_parameters_keys() == ["a", "b", "x", "y", "z"]


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


@pytest.mark.parametrize(
    "arrow_operator_supported, expected",
    [
        [False, expected_json_extract],
        [True, expected_arrow],
    ],
    ids=[
        "arrow-not-supported",
        "arrow-supported",
    ],
)
def test_get_sample_query(arrow_operator_supported, expected, monkeypatch):
    monkeypatch.setattr(
        tracker_module, "ARROW_OPERATOR_SUPPORTED", arrow_operator_supported
    )

    tracker = SQLiteTracker(":memory:")

    to_insert = [
        dict(a=1, b=2),
        dict(x=1, y=2),
        dict(z=3),
    ]

    for i, data in enumerate(to_insert):
        tracker.insert(i, data)

    assert tracker.get_sample_query(compatibility_mode=False) == expected


def test_experiment_log():
    tracker = SQLiteTracker(":memory:")
    experiment = tracker.new_experiment()
    experiment.log("accuracy", 0.8)

    retrieved = tracker.get(experiment.uuid)

    assert retrieved == {"accuracy": 0.8}
