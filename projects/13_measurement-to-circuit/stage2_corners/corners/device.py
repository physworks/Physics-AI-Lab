"""Resolve project 12's model and stage 1's analysis, without copying either.

Stage 2 consumes both: the compact model defines what a parameter *is*, and
stage 1's Fisher machinery supplies the extraction covariance this stage has
to subtract from the observed spread.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve()
_STAGE_ROOT = _HERE.parent.parent                   # .../stage2_corners
_PROJECT_ROOT = _STAGE_ROOT.parent                  # .../13_measurement-to-circuit
_STAGE1 = _PROJECT_ROOT / "stage1_design"
_CMEXT = _PROJECT_ROOT.parent / "12_compact-model-extraction"


def _locate() -> None:
    env = os.environ.get("CMEXT_PATH")
    for cand in (Path(env) if env else None, _CMEXT):
        if cand and (cand / "cmext" / "model.py").is_file():
            sys.path.insert(0, str(cand))
            break
    else:
        try:
            import cmext  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "project 12 (12_compact-model-extraction) was not found. "
                "Clone the full repository so it sits two levels up, or set "
                "CMEXT_PATH."
            ) from exc

    if (_STAGE1 / "oed" / "fisher.py").is_file():
        sys.path.insert(0, str(_STAGE1))
    else:                                            # pragma: no cover
        raise ImportError(
            "stage1_design was not found next to this stage. Stage 2 uses its "
            "Fisher analysis to separate extraction noise from process spread."
        )


_locate()

from cmext.model import DeviceParams, drain_current, measure  # noqa: E402
from oed.constraints import (DesignConstraints, build_pool,  # noqa: E402
                             default_required_regions)
from oed.device import extract_subset  # noqa: E402
from oed.fisher import information  # noqa: E402
from oed.noise import NoiseModel  # noqa: E402
from oed.objective import full_objective  # noqa: E402
from oed.select import greedy_design  # noqa: E402

PARAM_NAMES: List[str] = DeviceParams.names()

__all__ = ["DeviceParams", "PARAM_NAMES", "DesignConstraints", "NoiseModel",
           "build_pool", "default_required_regions", "drain_current",
           "extract_subset", "full_objective", "greedy_design", "information",
           "measure"]
