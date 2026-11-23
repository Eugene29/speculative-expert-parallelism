"""Integration tests for the pipelined combine-dispatch (spec_ep.pipeline).

Runs both pipelines end to end for every degree in DEGREES on every layout, pilot
rule and routing of tests/checks.py, which explains the check. A rank holds 16
tokens, so degrees 3 and 5 cut slices of unequal size.

Usage:
    python -m pytest tests/test_pipeline.py -m smoke   # 2_nodes only, for development
    python -m pytest tests/test_pipeline.py -v         # every layout
"""
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist

from checks import LAYOUTS, Layout, run_check
from spec_ep.pipeline import pipelined_plain_combine_dispatch, pipelined_rbd_combine_dispatch

DEGREES = (1, 2, 3, 5)


def _plain_pipeline(
    *args: Any,
    gpus_per_node: int,
    local_group: dist.ProcessGroup,
    pilot_rule: str,
    **kwargs: Any,
) -> torch.Tensor:
    """pipelined_plain_combine_dispatch, dropping the arguments only RBD takes.

    Returns:
        output: [num_tokens, H] output of the plain pipeline.
    """
    return pipelined_plain_combine_dispatch(*args, **kwargs)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_pipelines_match_local_reference(tmp_path: Path, layout: Layout) -> None:
    """Every pipeline gives the locally computed answer on every rank, for every degree."""
    variants = {}
    for degree in DEGREES:
        variants[f"rbd degree={degree}"] = partial(pipelined_rbd_combine_dispatch, degree=degree)
        variants[f"rbd two lanes degree={degree}"] = partial(
            pipelined_rbd_combine_dispatch, degree=degree, lanes=2
        )
        variants[f"plain degree={degree}"] = partial(_plain_pipeline, degree=degree)
    run_check(tmp_path, layout, variants)
