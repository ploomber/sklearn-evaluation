from sklearn_evaluation.nb import sets


def test_differences():
    both, a_only, b_only = sets.differences([1, 2, 3], [1, 4, 5])
    assert both == {1}
    assert a_only == {2, 3}
    assert b_only == {4, 5}
