from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def testdatadir():
    return Path().absolute() / "tests" / "testdata"
