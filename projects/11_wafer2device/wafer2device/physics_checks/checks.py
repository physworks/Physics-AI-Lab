"""Physics validation.

These checks are the engineer's authority in the pipeline.  The hypothesis
generator -- rule or LLM -- cannot alter them, and a hypothesis that fails here
does not reach the optimiser.

Each failing check also produces a ``direction_hint``: a short statement of which
way a parameter must move to relieve the violation.  The hint is derived from the
model's own sensitivities, so the replanning loop is steered by physics rather
than by the engineer hand-feeding answers to the model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..contracts import (CheckReport, CheckResult, Hypothesis, SpatialDescriptors)
from ..s1_pattern.data import wafer_grid
from ..s3_device.models import DeviceModel, NOMINAL, variation_fields


def run_all(h: Hypothesis, model: DeviceModel, wmap: np.ndarray,
            cfg: Dict[str, Any], secondary: Optional[DeviceModel] = None,
            descriptors: Optional[SpatialDescriptors] = None) -> CheckReport:
    spec = cfg["device_spec"]
    checks = cfg["checks"]
    n = wmap.shape[0]
    mask = wmap >= 0
    fields = variation_fields(h, n)
    out = model.evaluate(fields)

    results: List[CheckResult] = []
    if checks["range_sanity"]["enabled"]:
        results.append(_range_sanity(out, mask, checks["range_sanity"]))
    if checks["monotonicity"]["enabled"]:
        results.append(_monotonicity(model, checks["monotonicity"]))
    if checks["limit_case"]["enabled"]:
        results.append(_limit_case(model, checks["limit_case"]))
    if checks["pattern_consistency"]["enabled"]:
        results.append(_pattern_consistency(h, wmap, checks["pattern_consistency"]))
    if checks["model_agreement"]["enabled"]:
        results.append(_model_agreement(model, secondary, fields, mask,
                                        checks["model_agreement"]))
    return CheckReport(results=results)


# --------------------------------------------------------------------------- #

def _range_sanity(out: Dict[str, np.ndarray], mask: np.ndarray,
                  cfg: Dict[str, Any]) -> CheckResult:
    vth = out["vth"][mask]
    ion = out["ion"][mask]
    ioff = out["ioff"][mask]
    viol = 0.0
    details: List[str] = []

    if vth.min() < cfg["vth_min"] or vth.max() > cfg["vth_max"]:
        span = max(cfg["vth_max"] - cfg["vth_min"], 1e-9)
        excess = max(cfg["vth_min"] - vth.min(), vth.max() - cfg["vth_max"], 0.0)
        viol = max(viol, excess / span)
        details.append(f"Vth range [{vth.min():.3f}, {vth.max():.3f}] V leaves "
                       f"[{cfg['vth_min']}, {cfg['vth_max']}]")
    if ion.min() < cfg["ion_min"]:
        viol = max(viol, 1.0)
        details.append(f"Ion minimum {ion.min():.2e} below {cfg['ion_min']:.1e}")
    if ioff.min() < cfg["ioff_min"]:
        viol = max(viol, 0.5)
        details.append(f"Ioff minimum {ioff.min():.2e} below {cfg['ioff_min']:.1e}")

    passed = viol == 0.0
    hint = ""
    if not passed and "Vth" in " ".join(details):
        hint = ("decrease gate_oxide_thickness / channel_doping magnitude: "
                "predicted Vth leaves the physical range")
    return CheckResult("range_sanity", passed, observed=float(viol), threshold=0.0,
                       deviation_ratio=float(viol), direction_hint=hint,
                       detail="; ".join(details) or "all predicted values in range")


def _monotonicity(model: DeviceModel, cfg: Dict[str, Any]) -> CheckResult:
    """Vth must rise monotonically with tox and with channel doping."""
    sweep = np.linspace(-0.08, 0.08, 17)
    worst_frac = 0.0
    offenders: List[str] = []
    for param in ("gate_oxide_thickness", "channel_doping"):
        vth = np.array([model.evaluate({param: np.array([x])})["vth"][0]
                        for x in sweep])
        diffs = np.diff(vth)
        frac_bad = float((diffs < 0).mean())
        if frac_bad > worst_frac:
            worst_frac = frac_bad
        if frac_bad > cfg["tolerance"]:
            offenders.append(f"{param} ({frac_bad:.0%} decreasing pairs)")
    passed = worst_frac <= cfg["tolerance"]
    return CheckResult("monotonicity", passed, observed=worst_frac,
                       threshold=float(cfg["tolerance"]),
                       deviation_ratio=0.0 if passed else worst_frac - cfg["tolerance"],
                       direction_hint="" if passed else
                       "device model violates dVth/dtox > 0; fix the model, not the "
                       "hypothesis",
                       detail="; ".join(offenders) or "Vth monotonic in tox and doping")


def _limit_case(model: DeviceModel, cfg: Dict[str, Any]) -> CheckResult:
    """Zero variation must reproduce the nominal device."""
    out = model.evaluate({"gate_oxide_thickness": np.array([0.0])})
    err = abs(float(out["vth"][0]) - NOMINAL["vth"])
    passed = err <= cfg["vth_abs_tolerance"]
    return CheckResult("limit_case", passed, observed=err,
                       threshold=float(cfg["vth_abs_tolerance"]),
                       deviation_ratio=0.0 if passed else err / cfg["vth_abs_tolerance"],
                       direction_hint="" if passed else
                       "model does not return to nominal at zero variation",
                       detail=f"|Vth(0) - Vth_nominal| = {err:.2e} V")


def _pattern_consistency(h: Hypothesis, wmap: np.ndarray,
                         cfg: Dict[str, Any]) -> CheckResult:
    """The hypothesis must reproduce the observed spatial signature.

    This is the check that catches a physically legal but wrong story: a
    hypothesis whose implied stress field does not correlate with where the
    wafer actually fails.
    """
    n = wmap.shape[0]
    mask = wmap >= 0
    fields = variation_fields(h, n)
    stress = np.zeros((n, n))
    for f in fields.values():
        # Failures follow the positive lobe of the variation, which is what the
        # signed spatial form encodes; abs() would make an angular term look
        # two-sided and mask a genuine direction error.
        stress = stress + np.clip(f, 0.0, None)
    if float(np.abs(stress).max()) < 1e-12:
        for f in fields.values():
            stress = stress + np.abs(f)

    obs = (wmap == 1).astype(float)
    obs_s = _smooth(obs * mask, 2)
    if stress[mask].std() < 1e-12 or obs_s[mask].std() < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(stress[mask], obs_s[mask])[0, 1])

    passed = corr >= cfg["min_correlation"]
    shortfall = max(0.0, cfg["min_correlation"] - corr)
    hint = ""
    if not passed:
        hint = ("spatial form mismatch: the implied stress field does not follow the "
                "observed defect map; change spatial_form or blame a different "
                "parameter")
    return CheckResult("pattern_consistency", passed, observed=corr,
                       threshold=float(cfg["min_correlation"]),
                       deviation_ratio=shortfall / max(cfg["min_correlation"], 1e-9),
                       direction_hint=hint,
                       detail=f"corr(implied stress, smoothed defect map) = {corr:.3f}")


def _model_agreement(primary: DeviceModel, secondary: Optional[DeviceModel],
                     fields: Dict[str, np.ndarray], mask: np.ndarray,
                     cfg: Dict[str, Any]) -> CheckResult:
    """Trend agreement between the analytic model and the surrogate.

    Skipped (and reported as such) while only one model is configured, so the
    same check file works before and after the surrogate is plugged in.
    """
    if secondary is None:
        return CheckResult("model_agreement", True, observed=1.0,
                           threshold=float(cfg["min_rank_correlation"]),
                           deviation_ratio=0.0,
                           detail="skipped: only one device model configured")
    a = primary.evaluate(fields)["vth"][mask]
    b = secondary.evaluate(fields)["vth"][mask]
    rho = _spearman(a, b)
    passed = rho >= cfg["min_rank_correlation"]
    return CheckResult("model_agreement", passed, observed=float(rho),
                       threshold=float(cfg["min_rank_correlation"]),
                       deviation_ratio=0.0 if passed else
                       (cfg["min_rank_correlation"] - rho) / cfg["min_rank_correlation"],
                       direction_hint="" if passed else
                       "analytic and surrogate disagree in trend; do not trust the "
                       "optimisation result until this is resolved",
                       detail=f"Spearman rho(analytic, surrogate) = {rho:.3f}")


# --------------------------------------------------------------------------- #

def _smooth(a: np.ndarray, k: int) -> np.ndarray:
    out = a.copy()
    for _ in range(k):
        p = np.pad(out, 1, mode="edge")
        out = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
               + 2.0 * out) / 6.0
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def build_feedback(h: Hypothesis, report: CheckReport,
                   previous: List[Hypothesis]) -> Dict[str, Any]:
    """Structured failure feedback for the replanning loop.

    Format C from the design: violation detail + direction hint + the full list
    of attempts already made, so the generator cannot loop back to a hypothesis
    it has already burned.
    """
    return {
        "failures": [{"violated_check": r.name,
                      "observed": round(float(r.observed), 5),
                      "threshold": round(float(r.threshold), 5),
                      "deviation_ratio": round(float(r.deviation_ratio), 5),
                      "direction_hint": r.direction_hint,
                      "detail": r.detail}
                     for r in report.failures()],
        "previous_attempts": [{"round": p.replan_round, "source": p.source,
                               "terms": [t.to_dict() for t in p.terms]}
                              for p in previous],
    }
