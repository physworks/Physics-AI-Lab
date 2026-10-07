"""Shared fixtures.

The 24 fault cases cost about 40 seconds to solve, almost all of it in the
singular cases that run to the iteration limit while the solver computes a
condition number and an eigenvalue spread every step.  They are solved once
per session and shared; no test is allowed to mutate a trace.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from circuit.device import DeviceParams          # noqa: E402
from circuit.faults import Case, build_cases, solve_case   # noqa: E402


@pytest.fixture(scope="session")
def nominal() -> DeviceParams:
    return DeviceParams()


@pytest.fixture(scope="session")
def cases() -> List[Case]:
    return build_cases(seed=0)


@pytest.fixture(scope="session")
def traces(cases) -> Dict[str, Dict]:
    return {c.name: solve_case(c) for c in cases}


@pytest.fixture(scope="session")
def truths(cases) -> Dict[str, str]:
    return {c.name: c.truth for c in cases}
