import subprocess
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def build_deterministic_dataset() -> None:
    """Build the Parquet fixture so a fresh clone can run pytest directly."""
    subprocess.run([sys.executable, "scripts/build_sample_data.py"], check=True)
