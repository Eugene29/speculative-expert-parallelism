"""Integration tests for Redundancy-Bypassing Dispatch (spec_ep.rbd).

Runs rbd_combine_dispatch end to end on every layout, pilot rule and routing of
tests/checks.py, which explains the check.

Usage:
    python -m pytest tests/test_rbd.py -m smoke   # 2_nodes only, for development
    python -m pytest tests/test_rbd.py -v         # every layout
"""
from pathlib import Path

import pytest

from checks import LAYOUTS, Layout, run_check
from spec_ep.rbd import rbd_combine_dispatch


@pytest.mark.parametrize("layout", LAYOUTS)
def test_rbd_matches_local_reference(tmp_path: Path, layout: Layout) -> None:
    """RBD gives the locally computed answer on every rank."""
    run_check(tmp_path, layout, {"rbd": rbd_combine_dispatch})
