"""Integration tests for two-lane RBD (spec_ep.two_lane).

Runs two_lane_rbd_combine_dispatch end to end on every layout, pilot rule and routing
of tests/checks.py, which explains the check.

Usage:
    python -m pytest tests/test_two_lane.py -m smoke   # 2_nodes only, for development
    python -m pytest tests/test_two_lane.py -v         # every layout
"""
from functools import partial
from pathlib import Path

import pytest

from checks import LAYOUTS, Layout, run_check
from spec_ep.two_lane import two_lane_rbd_combine_dispatch


@pytest.mark.parametrize("layout", LAYOUTS)
def test_two_lane_matches_local_reference(tmp_path: Path, layout: Layout) -> None:
    """Two-lane RBD gives the locally computed answer on every rank."""
    run_check(tmp_path, layout, {
        "two_lane": two_lane_rbd_combine_dispatch,
        **{f"two_lane early={early}": partial(two_lane_rbd_combine_dispatch, early=early)
           for early in (0.0, 0.5, 1.0)},
    })
