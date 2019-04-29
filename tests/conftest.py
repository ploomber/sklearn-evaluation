from pathlib import Path

import pytest
from matplotlib.testing.conftest import (mpl_test_settings,
                                         mpl_image_comparison_parameters,
                                         pytest_configure, pytest_unconfigure,
                                         pd)


def _path_to_tests():
    return Path(__file__).absolute().parent


@pytest.fixture(scope='session')
def path_to_tests():
    return _path_to_tests()
