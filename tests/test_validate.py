import pytest
from sklearn_evaluation import validate


@pytest.mark.parametrize('valid', [0, 1, 0.1, 0.5, 0.8, 0.9999])
def test_decorator_argument_is_proportion(valid):
    def fn(proportion):
        pass

    validate.argument_is_proportion('proportion')(fn)(proportion=valid)


@pytest.mark.parametrize('invalid', [-1, -0.5, -0.001, 1.00001, 1.1])
def test_decorator_argument_is_proportion_error(invalid):
    def fn(proportion):
        pass

    with pytest.raises(ValueError):
        validate.argument_is_proportion('proportion')(fn)(proportion=invalid)
