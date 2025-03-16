'''
Test fixtures
'''
import pytest

from pathlib import Path

@pytest.fixture
def outdir():
    dir = Path(__file__).parent / 'out'
    dir.mkdir(exist_ok=True)
    return dir
