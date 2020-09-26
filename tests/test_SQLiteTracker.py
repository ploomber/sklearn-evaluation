import pytest
from sklearn_evaluation.manage.SQLiteTracker import SQLiteTracker


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
    res = tracker.get(uuid)

    assert res.loc[uuid].comment == 'this is a comment'


def test_get():
    tracker = SQLiteTracker(':memory:')
    tracker.insert('some_uuid', {'a': 1})
    res = tracker.get('some_uuid')
    content = res.loc['some_uuid'].content

    assert len(res) == 1
    assert content == '{"a": 1}'


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
    res = tracker.get(uuid)
    assert len(res) == 1
