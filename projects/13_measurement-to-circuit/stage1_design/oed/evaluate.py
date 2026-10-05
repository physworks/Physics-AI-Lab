"""Scoring the arms on something none of them optimised.

Each arm optimises its own criterion, so comparing arms on any of those
criteria would hand the win to whoever was asked the question.  The primary
metric here is deliberately outside every arm's objective:

    **held-out prediction error** -- fit at the designed bias points, then
    predict the noise-free current on a dense grid fixed in the configuration
    and never shown to any design.

That is also the metric a modelling organisation actually cares about: a
parameter set is only worth what it predicts at bias points nobody measured.
Parameter recovery and the Fisher-predicted sigma are reported alongside, but
they are secondary and one of them is an arm's own objective.

The Monte Carlo loop additionally re-checks the Fisher prediction against the
realised spread, the same way project 12 does, because a design chosen by a
Fisher matrix is only as good as that matrix's honesty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .device import (DeviceParams, drain_current, extract_subset,
                     measure)
from .fisher import information
from .noise import NoiseModel
from .select import Design


@dataclass
class ArmResult:
    arm: str
    n_points: int
    predict_rms_log10: float
    predict_rms_std: float
    param_rel_error: Dict[str, float] = field(default_factory=dict)
    realised_rel_sigma: Dict[str, float] = field(default_factory=dict)
    predicted_rel_sigma: Dict[str, float] = field(default_factory=dict)
    free_params: List[str] = field(default_factory=list)
    failures: int = 0
    seeds: int = 0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> Dict:
        return {"arm": self.arm, "n_points": self.n_points,
                "predict_rms_log10": self.predict_rms_log10,
                "predict_rms_std": self.predict_rms_std,
                "free_params": list(self.free_params),
                "param_rel_error": dict(self.param_rel_error),
                "realised_rel_sigma": dict(self.realised_rel_sigma),
                "predicted_rel_sigma": dict(self.predicted_rel_sigma),
                "failures": self.failures, "seeds": self.seeds,
                "notes": list(self.notes)}


def holdout_grid(vgs_range=(0.05, 1.2), n_vgs: int = 40,
                 vds_list: Sequence[float] = (0.05, 0.3, 0.6, 0.9, 1.2)):
    """The dense grid every arm is judged on.  Never used to choose points."""
    vg_axis = np.linspace(vgs_range[0], vgs_range[1], n_vgs)
    vg, vd = np.meshgrid(vg_axis, np.asarray(vds_list, dtype=float),
                         indexing="ij")
    return vg.ravel(), vd.ravel()


def prediction_error(truth: DeviceParams, est: DeviceParams,
                     vg_h, vd_h, floor: float = 1e-12) -> float:
    """RMS of log10 current error on the held-out grid, above the floor."""
    i_true = drain_current(truth, vg_h, vd_h)
    i_est = drain_current(est, vg_h, vd_h)
    ok = (i_true > 3.0 * floor) & (i_est > 0)
    if not np.any(ok):
        return float("nan")
    return float(np.sqrt(np.mean(
        (np.log10(i_est[ok]) - np.log10(i_true[ok])) ** 2)))


def run_arm(truth: DeviceParams, design: Design, noise: NoiseModel,
            seeds: int = 40, holdout=None,
            noise_floor: float = 1e-12) -> ArmResult:
    """Monte Carlo over noise realisations for one design."""
    if holdout is None:
        holdout = holdout_grid()
    vg_h, vd_h = holdout

    free = list(design.objective.include_params)
    fixed = dict(design.objective.fixed_params)

    errs: List[float] = []
    ests: Dict[str, List[float]] = {n: [] for n in free}
    failures = 0

    for s in range(seeds):
        ids = measure(truth, design.vgs, design.vds,
                      noise_rel=noise.sigma_rel, noise_floor=noise.i_floor,
                      seed=10_000 + s)
        try:
            est = extract_subset(design.vgs, design.vds, ids, free, fixed,
                                 noise_floor=noise_floor)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            failures += 1
            continue
        e = prediction_error(truth, est, vg_h, vd_h, noise_floor)
        if not np.isfinite(e):
            failures += 1
            continue
        errs.append(e)
        for n in free:
            ests[n].append(getattr(est, n))

    truth_d = truth.to_dict()
    rel_err, realised = {}, {}
    for n in free:
        a = np.array(ests[n], dtype=float)
        if a.size < 2:
            continue
        rel_err[n] = float(abs(a.mean() - truth_d[n])
                           / max(abs(truth_d[n]), 1e-30))
        realised[n] = float(a.std() / max(abs(a.mean()), 1e-30))

    fisher = information(truth, design.vgs, design.vds, noise, free)

    notes = []
    if failures:
        notes.append(f"{failures} of {seeds} extractions did not complete")

    return ArmResult(
        arm=design.arm, n_points=len(design.indices),
        predict_rms_log10=float(np.mean(errs)) if errs else float("nan"),
        predict_rms_std=float(np.std(errs)) if errs else float("nan"),
        param_rel_error=rel_err, realised_rel_sigma=realised,
        predicted_rel_sigma=dict(fisher.rel_sigma),
        free_params=free, failures=failures, seeds=seeds, notes=notes)


def equivalent_points(curve: Dict[int, float], target: float) -> int | None:
    """Fewest points at which a curve reaches ``target`` (lower is better).

    ``curve`` maps point count to held-out prediction error.  The headline
    number of this stage is the answer to "how few points does this reach the
    uniform grid's accuracy with", so it is read off exactly this way rather
    than interpolated: an integer count is what a measurement plan buys.
    """
    for n in sorted(curve):
        v = curve[n]
        if np.isfinite(v) and v <= target:
            return int(n)
    return None
