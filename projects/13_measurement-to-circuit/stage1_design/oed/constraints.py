"""The candidate bias points, and the rules a design must obey.

Without constraints this problem is textbook: log det of a Fisher matrix is
submodular in the set of measurements, so greedy selection carries a
(1 - 1/e) guarantee and in practice lands very close to optimal.  Claiming
that anything beats greedy on an unconstrained D-optimal objective would be
wrong, and this module is where that stops being the problem being solved.

Three things break the clean version, all of them real:

  * **Stress.**  High gate and drain bias together heat and degrade the
    device.  A point that would be informative is simply not allowed.
  * **Settling.**  Changing the drain source costs far more time than
    stepping the gate, so the number of *distinct* drain biases is capped.
    Whether a point is cheap now depends on what else was already chosen,
    which is precisely the structure greedy assumes away.
  * **Specification.**  Some points must be measured whatever the
    information content: off-state leakage and on-current are reported
    regardless of what the model needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .device import DeviceParams, drain_current


@dataclass(frozen=True)
class CandidatePool:
    """Every bias point the instrument could visit."""

    vgs: np.ndarray
    vds: np.ndarray
    current: np.ndarray
    allowed: np.ndarray            # bool, False where a constraint forbids it

    def __len__(self) -> int:
        return int(self.vgs.size)

    def allowed_index(self) -> np.ndarray:
        return np.flatnonzero(self.allowed)

    def take(self, idx: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(list(idx), dtype=int)
        return self.vgs[idx], self.vds[idx]

    def distinct_vds(self, idx: Sequence[int]) -> int:
        if len(idx) == 0:
            return 0
        return int(np.unique(np.round(self.vds[np.asarray(list(idx),
                                                          dtype=int)], 9)).size)


@dataclass
class DesignConstraints:
    """Hard limits.  Nothing in the pipeline, agent included, may relax these."""

    n_points: int = 12
    max_distinct_vds: int = 3
    power_limit_w: float = 2.0e-4          # I*Vd ceiling, self-heating
    min_snr: float = 3.0                   # multiples of the noise floor
    required_regions: List[Dict] = field(default_factory=list)

    def describe(self) -> Dict:
        return {"n_points": self.n_points,
                "max_distinct_vds": self.max_distinct_vds,
                "power_limit_w": self.power_limit_w,
                "min_snr": self.min_snr,
                "required_regions": [dict(r) for r in self.required_regions]}


def default_required_regions() -> List[Dict]:
    """Points the process reports whether or not the model wants them."""
    return [
        {"name": "off_state", "vgs_max": 0.10, "count": 1,
         "why": "leakage is a reported specification"},
        {"name": "on_state", "vgs_min": 1.10, "count": 1,
         "why": "drive current is a reported specification"},
    ]


def build_pool(p: DeviceParams, constraints: DesignConstraints,
               vgs_range: Tuple[float, float] = (0.0, 1.2), n_vgs: int = 25,
               vds_list: Sequence[float] = (0.05, 0.3, 0.6, 0.9, 1.2),
               noise_floor: float = 1e-12) -> CandidatePool:
    """Grid of candidate points with the infeasible ones marked."""
    vg_axis = np.linspace(vgs_range[0], vgs_range[1], n_vgs)
    vg, vd = np.meshgrid(vg_axis, np.asarray(vds_list, dtype=float),
                         indexing="ij")
    vg, vd = vg.ravel(), vd.ravel()

    current = drain_current(p, vg, vd)
    power_ok = current * vd <= constraints.power_limit_w
    snr_ok = current >= constraints.min_snr * noise_floor

    return CandidatePool(vgs=vg, vds=vd, current=current,
                         allowed=power_ok & snr_ok)


def region_members(pool: CandidatePool, region: Dict) -> np.ndarray:
    """Indices of allowed candidates satisfying one required region."""
    ok = pool.allowed.copy()
    if "vgs_min" in region:
        ok &= pool.vgs >= region["vgs_min"]
    if "vgs_max" in region:
        ok &= pool.vgs <= region["vgs_max"]
    if "vds_min" in region:
        ok &= pool.vds >= region["vds_min"]
    if "vds_max" in region:
        ok &= pool.vds <= region["vds_max"]
    return np.flatnonzero(ok)


def feasible_addition(pool: CandidatePool, chosen: Sequence[int],
                      candidate: int,
                      constraints: DesignConstraints) -> bool:
    """May ``candidate`` be added to ``chosen`` without breaking a rule?"""
    if not pool.allowed[candidate] or candidate in chosen:
        return False
    if len(chosen) >= constraints.n_points:
        return False
    return pool.distinct_vds(list(chosen) + [candidate]) \
        <= constraints.max_distinct_vds


def violations(pool: CandidatePool, chosen: Sequence[int],
               constraints: DesignConstraints) -> List[str]:
    """Every rule a finished design breaks.  Empty means acceptable."""
    out: List[str] = []
    chosen = list(chosen)

    if len(chosen) != constraints.n_points:
        out.append(f"{len(chosen)} points selected, {constraints.n_points} "
                   "required")
    if len(set(chosen)) != len(chosen):
        out.append("a bias point is selected more than once")
    if any(not pool.allowed[i] for i in chosen):
        out.append("a selected point violates the power or SNR limit")

    nvd = pool.distinct_vds(chosen)
    if nvd > constraints.max_distinct_vds:
        out.append(f"{nvd} distinct drain biases, limit is "
                   f"{constraints.max_distinct_vds}")

    for region in constraints.required_regions:
        members = set(region_members(pool, region).tolist())
        have = len(members & set(chosen))
        need = int(region.get("count", 1))
        if have < need:
            out.append(f"required region {region['name']!r}: {have} of {need} "
                       f"({region.get('why', '')})")
    return out
