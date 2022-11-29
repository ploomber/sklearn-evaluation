"""
Test configuration for doctests
"""
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def tmp_empty(tmp_path):
    """
    Create temporary path using pytest native fixture,
    them move it, yield, and restore the original path
    """
    old = os.getcwd()
    os.chdir(str(tmp_path))
    yield str(Path(tmp_path).resolve())
    os.chdir(old)
