"""Bridge to the compact model and extractor of project 12.

Stage 1 deliberately does not re-implement the device model.  The question
here is *which measurements to take*, and answering it honestly requires the
same model and the same extractor that project 12 already validated -- a
second, subtly different copy would make every comparison meaningless.

Resolution order for project 12:
  1. the ``CMEXT_PATH`` environment variable
  2. ``../../12_compact-model-extraction`` relative to this project
  3. an already-importable ``cmext`` package
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

_HERE = Path(__file__).resolve()
_STAGE_ROOT = _HERE.parent.parent              # .../stage1_design
_PROJECT_ROOT = _STAGE_ROOT.parent             # .../13_measurement-to-circuit
_SIBLING = _PROJECT_ROOT.parent / "12_compact-model-extraction"


def _locate() -> None:
    env = os.environ.get("CMEXT_PATH")
    for cand in (Path(env) if env else None, _SIBLING):
        if cand and (cand / "cmext" / "model.py").is_file():
            sys.path.insert(0, str(cand))
            return
    try:
        import cmext  # noqa: F401
    except ImportError as exc:                 # pragma: no cover
        raise ImportError(
            "project 12 (12_compact-model-extraction) was not found.\n"
            "It provides the compact model and the extractor this stage "
            "builds on.  Either clone the full Physics-AI-Lab repository so "
            "that 12_compact-model-extraction sits next to this project, or "
            "set CMEXT_PATH to its directory."
        ) from exc


_locate()

from cmext.extraction import (_fit_subset, _log_residual,  # noqa: E402
                              _usable, seed_params)
from cmext.identifiability import log_jacobian  # noqa: E402
from cmext.model import (BOUNDS, DeviceParams, drain_current,  # noqa: E402
                         measure)

__all__ = ["BOUNDS", "DeviceParams", "drain_current", "measure",
           "log_jacobian", "seed_params", "extract_subset", "PARAM_NAMES"]

PARAM_NAMES: List[str] = DeviceParams.names()


def extract_subset(vgs, vds, ids, free: Sequence[str],
                   fixed: Dict[str, float] | None = None,
                   noise_floor: float = 1e-12) -> DeviceParams:
    """Fit only ``free``; hold everything else at ``fixed`` (or the seed).

    A design that decides a parameter is not identifiable has to act on that
    decision at extraction time too, otherwise the unidentifiable parameter
    still absorbs variance and the design choice means nothing.  This is the
    extraction counterpart of dropping a parameter from the objective.
    """
    vgs = np.asarray(vgs, dtype=float)
    vds = np.asarray(vds, dtype=float)
    ids = np.asarray(ids, dtype=float)

    base = seed_params()
    if fixed:
        base = base.copy_with(**fixed)

    free = list(free)
    unknown = set(free) - set(PARAM_NAMES)
    if unknown:
        raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
    if not free:
        raise ValueError("at least one parameter must be free")

    mask = _usable(ids, noise_floor)
    if mask.sum() < len(free) + 1:
        raise ValueError(
            f"{int(mask.sum())} usable points for {len(free)} free parameters")

    vals = _fit_subset(base, free, mask, vgs, vds, ids, base.to_dict())
    return base.copy_with(**vals)


def log_residual_rms(p: DeviceParams, vgs, vds, ids,
                     noise_floor: float = 1e-12) -> float:
    """RMS of log10 current residual over the usable points."""
    ids = np.asarray(ids, dtype=float)
    mask = _usable(ids, noise_floor)
    r = _log_residual(p, np.asarray(vgs)[mask], np.asarray(vds)[mask],
                      ids[mask])
    return float(np.sqrt(np.mean(r ** 2)))
