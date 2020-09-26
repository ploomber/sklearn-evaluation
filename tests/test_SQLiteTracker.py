from uuid import uuid4

import pytest
from sklearn_evaluation.SQLiteTracker import SQLiteTracker


def test_insert():
    tracker = SQLiteTracker(':memory:')
    tracker.insert('some_uuid', {'a': 1})

    res = tracker.query('SELECT * FROM experiments WHERE uuid = "some_uuid"')

    assert len(res) == 1


def test_new():
    tracker = SQLiteTracker(':memory:')
    assert isinstance(tracker.new(), str)


def test_comment():
    tracker = SQLiteTracker(':memory:')
    uuid = tracker.new()
    tracker.comment(uuid, 'this is a comment')
    res = tracker[uuid]

    assert res.loc[uuid].comment == 'this is a comment'


def test_get():
    tracker = SQLiteTracker(':memory:')
    tracker.insert('some_uuid', {'a': 1})
    res = tracker['some_uuid']
    parameters = res.loc['some_uuid'].parameters

    assert len(res) == 1
    assert parameters == '{"a": 1}'


def test_update_errors():
    tracker = SQLiteTracker(':memory:')

    with pytest.raises(ValueError) as excinfo:
        tracker.update('unknown_uuid', {})

    assert 'not exist' in str(excinfo.value)

    tracker.insert('some_uuid', {})

    with pytest.raises(ValueError) as excinfo:
        tracker.update('some_uuid', {})

    assert 'non-empty' in str(excinfo.value)


def test_update():
    tracker = SQLiteTracker(':memory:')
    uuid = tracker.new()
    tracker.update(uuid, {})
    res = tracker[uuid]
    assert len(res) == 1


def test_reprs():
    tracker = SQLiteTracker(':memory:')

    assert 'SQLiteTracker' in repr(tracker)
    assert 'SQLiteTracker' in tracker._repr_html_()
    assert '(No experiments saved yet)' in repr(tracker)
    assert '(No experiments saved yet)' in tracker._repr_html_()

    uuids = [uuid4().hex for _ in range(6)]

    for i, uuid in enumerate(uuids):
        tracker.insert(uuid, {'a': i})

    expected = [True, True, True, True, True, False]

    assert [uuid in repr(tracker) for uuid in uuids] == expected
    assert [uuid in tracker._repr_html_() for uuid in uuids] == expected


def test_recent():
    tracker = SQLiteTracker(':memory:')

    for i in range(5):
        tracker.insert(i, {'a': i})

    df = tracker.recent(normalize=False)

    assert df.columns.tolist() == ['created', 'parameters', 'comment']
    assert df.index.name == 'uuid'

    df = tracker.recent(normalize=True)

    assert df.columns.tolist() == ['created', 'a', 'comment']
    assert df.index.name == 'uuid'
