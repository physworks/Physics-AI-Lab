"""Turning an objective into a set of bias points.

Every arm in this project uses the *same* selector.  That is deliberate: the
comparison is between objectives, not between optimisers.  If one arm also
got a better search, a win would say nothing about the decision that arm made.

The selector is constrained greedy, seeded with the specification points that
must be measured regardless.  Greedy is the right baseline here rather than a
weak strawman -- log det is submodular, so on the unconstrained problem greedy
is already near optimal.  The constraints in ``constraints.py`` are what make
the objective choice matter more than the search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .constraints import (CandidatePool, DesignConstraints, feasible_addition,
                          region_members, violations)
from .device import DeviceParams
from .fisher import JacobianCache, build_cache, evaluate_design, information
from .noise import NoiseModel
from .objective import Objective


@dataclass
class Design:
    arm: str
    indices: List[int]
    vgs: np.ndarray
    vds: np.ndarray
    objective: Objective
    score: float
    violations: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> Dict:
        return {"arm": self.arm, "n_points": len(self.indices),
                "distinct_vds": sorted(set(np.round(self.vds, 4).tolist())),
                "objective": self.objective.describe(),
                "score": self.score, "violations": list(self.violations),
                "notes": list(self.notes)}


def _seed_required(pool: CandidatePool, constraints: DesignConstraints,
                   cache: JacobianCache,
                   objective: Objective) -> List[int]:
    """Satisfy the specification regions first, most informative point each."""
    chosen: List[int] = []
    for region in constraints.required_regions:
        members = [i for i in region_members(pool, region).tolist()
                   if i not in chosen]
        need = int(region.get("count", 1))
        for _ in range(need):
            best, best_s = None, -np.inf
            for i in members:
                if i in chosen or not feasible_addition(pool, chosen, i,
                                                        constraints):
                    continue
                s = cache.score_subset(chosen + [i], objective.include_params,
                                       objective.criterion, objective.c_target)
                if s > best_s:
                    best, best_s = i, s
            if best is not None:
                chosen.append(best)
    return chosen


def greedy_design(p: DeviceParams, pool: CandidatePool,
                  constraints: DesignConstraints, noise: NoiseModel,
                  objective: Objective, arm: str = "greedy",
                  cache: JacobianCache | None = None) -> Design:
    """Constrained greedy maximisation of the objective's score."""
    if cache is None:
        cache = build_cache(p, pool.vgs, pool.vds, noise)
    chosen = _seed_required(pool, constraints, cache, objective)
    notes = [f"{len(chosen)} point(s) fixed by specification regions"]

    while len(chosen) < constraints.n_points:
        best, best_s = None, -np.inf
        for i in pool.allowed_index().tolist():
            if not feasible_addition(pool, chosen, i, constraints):
                continue
            s = cache.score_subset(chosen + [i], objective.include_params,
                                   objective.criterion, objective.c_target)
            if s > best_s:
                best, best_s = i, s
        if best is None:
            notes.append("no feasible addition remained; design is short")
            break
        chosen.append(best)

    vg, vd = pool.take(chosen)
    final, _res = evaluate_design(p, vg, vd, noise, objective.include_params,
                                  objective.criterion, objective.c_target)
    return Design(arm=arm, indices=chosen, vgs=vg, vds=vd, objective=objective,
                  score=final,
                  violations=violations(pool, chosen, constraints),
                  notes=notes)


def uniform_design(p: DeviceParams, pool: CandidatePool,
                   constraints: DesignConstraints, noise: NoiseModel,
                   objective: Objective,
                   vds_used: Sequence[float] = (0.05, 1.2)) -> Design:
    """The status quo: evenly spaced gate bias at a couple of drain biases.

    This is what an evenly spaced measurement plan looks like when nobody has
    asked which points carry information -- the baseline a real line starts
    from, and the one the headline number is quoted against.
    """
    vds_used = list(vds_used)[:constraints.max_distinct_vds]
    per = constraints.n_points // len(vds_used)
    extra = constraints.n_points - per * len(vds_used)

    chosen: List[int] = []
    for k, vd_target in enumerate(vds_used):
        want = per + (1 if k < extra else 0)
        on_rail = np.flatnonzero(
            np.isclose(pool.vds, vd_target) & pool.allowed)
        if on_rail.size == 0:
            continue
        vg_vals = pool.vgs[on_rail]
        targets = np.linspace(vg_vals.min(), vg_vals.max(), want)
        for t in targets:
            order = np.argsort(np.abs(vg_vals - t))
            for j in order:
                cand = int(on_rail[j])
                if cand not in chosen:
                    chosen.append(cand)
                    break

    vg, vd = pool.take(chosen)
    s, _r = evaluate_design(p, vg, vd, noise, objective.include_params,
                            objective.criterion, objective.c_target)
    return Design(arm="uniform", indices=chosen, vgs=vg, vds=vd,
                  objective=objective, score=s,
                  violations=violations(pool, chosen, constraints),
                  notes=["evenly spaced gate bias; no information criterion"])


_REGIONS = (("subthreshold", 0.00, 0.34), ("threshold", 0.34, 0.52),
            ("linear", 0.52, 0.86), ("saturation", 0.86, 1.20))


def heuristic_design(p: DeviceParams, pool: CandidatePool,
                     constraints: DesignConstraints, noise: NoiseModel,
                     objective: Objective,
                     share: Sequence[float] = (0.25, 0.20, 0.30, 0.25),
                     ) -> Design:
    """What an experienced engineer writes down without computing anything.

    A fixed allocation across operating regions, split over the available
    drain biases.  It encodes the same instinct the staged extraction does:
    each region determines different parameters, so measure all of them.
    """
    vds_used = sorted(set(np.round(pool.vds, 9).tolist()))
    step = max(len(vds_used) // max(constraints.max_distinct_vds, 1), 1)
    vds_used = vds_used[::step][:constraints.max_distinct_vds]

    counts = [max(1, int(round(f * constraints.n_points))) for f in share]
    while sum(counts) > constraints.n_points:
        counts[int(np.argmax(counts))] -= 1
    while sum(counts) < constraints.n_points:
        counts[int(np.argmin(counts))] += 1

    chosen: List[int] = []
    for (name, lo, hi), want in zip(_REGIONS, counts):
        band = np.flatnonzero(
            pool.allowed & (pool.vgs >= lo) & (pool.vgs < hi)
            & np.isin(np.round(pool.vds, 9), vds_used))
        if band.size == 0:
            continue
        vg_vals = pool.vgs[band]
        for t in np.linspace(vg_vals.min(), vg_vals.max(), want):
            for j in np.argsort(np.abs(vg_vals - t)):
                cand = int(band[j])
                if cand not in chosen and feasible_addition(
                        pool, chosen, cand, constraints):
                    chosen.append(cand)
                    break

    vg, vd = pool.take(chosen)
    s, _r = evaluate_design(p, vg, vd, noise, objective.include_params,
                            objective.criterion, objective.c_target)
    return Design(arm="heuristic", indices=chosen, vgs=vg, vds=vd,
                  objective=objective, score=s,
                  violations=violations(pool, chosen, constraints),
                  notes=["fixed allocation across operating regions"])


def design_uncertainty(p: DeviceParams, design: Design, noise: NoiseModel,
                       params: Sequence[str]):
    """Predicted relative 1-sigma for ``params`` under this design."""
    return information(p, design.vgs, design.vds, noise, params)
