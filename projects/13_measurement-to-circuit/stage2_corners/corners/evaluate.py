"""Coverage against waste -- the only honest way to score a corner set.

A corner set defines a predicted range for a metric.  Two things can be wrong
with it, and they pull in opposite directions:

  **Coverage** -- the fraction of real devices whose metric falls inside the
  predicted range.  Below the nominal level the corners are unsafe: silicon
  exists that the designer never simulated.

  **Waste** -- how much wider the predicted range is than the range the
  devices actually occupy.  This is design margin spent on nothing.

Neither number decides anything alone.  A corner set covering 100% of devices
at four times the necessary width is not better than one covering 99.7% at
1.1x, and quoting only the first is how over-wide corners survive review.

The reference range is the population's own empirical quantiles at the level
the corners claim: +-3 sigma means 0.135% and 99.865%.  Quantiles rather than
mean +- 3s, because the metric distributions here are not symmetric -- current
is log-normal-ish when its parameters are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from typing import Dict, List

import numpy as np

from .generate import CornerSet
from .metrics import MetricSpec, evaluate_all


def nominal_coverage(k: float) -> float:
    """Two-sided Gaussian mass within k sigma."""
    return float(erf(k / sqrt(2.0)))


@dataclass
class MetricResult:
    metric: str
    method: str
    k: float
    predicted: tuple[float, float]
    reference: tuple[float, float]
    coverage: float
    nominal_coverage: float
    waste: float
    n_corners: int
    notes: List[str] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        """Does it cover what it claims to, allowing for sampling error?"""
        return self.coverage >= self.nominal_coverage - 0.005

    def summary(self) -> Dict:
        return {"metric": self.metric, "method": self.method, "k": self.k,
                "predicted_lo": self.predicted[0],
                "predicted_hi": self.predicted[1],
                "reference_lo": self.reference[0],
                "reference_hi": self.reference[1],
                "coverage": self.coverage,
                "nominal_coverage": self.nominal_coverage,
                "waste": self.waste, "safe": self.safe,
                "n_corners": self.n_corners, "notes": list(self.notes)}


def reference_range(values: np.ndarray, k: float) -> tuple[float, float]:
    tail = (1.0 - nominal_coverage(k)) / 2.0
    lo, hi = np.quantile(values, [tail, 1.0 - tail])
    return float(lo), float(hi)


def _width(lo: float, hi: float, log_scale: bool) -> float:
    if log_scale:
        return float(np.log10(max(hi, 1e-300)) - np.log10(max(lo, 1e-300)))
    return float(hi - lo)


def evaluate(corner_set: CornerSet, population_devices, metric: MetricSpec,
             k: float | None = None,
             truth: np.ndarray | None = None,
             corner_vals: np.ndarray | None = None) -> MetricResult:
    """Score one corner set against one metric on the true population.

    ``truth`` and ``corner_vals`` may be passed in already computed; the
    population's metrics do not change between corner sets, and recomputing
    them per set is most of the runtime.
    """
    k = corner_set.k if k is None else k

    if truth is None:
        truth = metric.evaluate(population_devices)
    if corner_vals is None:
        corner_vals = metric.evaluate(corner_set.as_devices())
    pred = (float(np.min(corner_vals)), float(np.max(corner_vals)))
    ref = reference_range(truth, k)

    inside = float(np.mean((truth >= pred[0]) & (truth <= pred[1])))
    w_pred = _width(*pred, metric.log_scale)
    w_ref = _width(*ref, metric.log_scale)
    waste = float(w_pred / w_ref - 1.0) if w_ref > 0 else float("inf")

    notes = []
    if corner_set.method == "statistical_mc":
        notes.append("a sample, not a corner set: its range is bounded by "
                     "what it happened to draw")
    return MetricResult(metric=metric.name, method=corner_set.method, k=k,
                        predicted=pred, reference=ref, coverage=inside,
                        nominal_coverage=nominal_coverage(k), waste=waste,
                        n_corners=len(corner_set), notes=notes)


def compare(corner_sets: Dict[str, CornerSet], population_devices,
            metrics: Dict[str, MetricSpec]) -> List[MetricResult]:
    """Every corner set against every metric, each population evaluated once."""
    truth = evaluate_all(population_devices)
    per_set = {name: evaluate_all(cs.as_devices())
               for name, cs in corner_sets.items()}

    out = []
    for m in metrics.values():
        for name, cs in corner_sets.items():
            out.append(evaluate(cs, population_devices, m,
                                truth=truth[m.name],
                                corner_vals=per_set[name][m.name]))
    return out


def best_by_waste(results: List[MetricResult], metric: str) -> MetricResult | None:
    """Narrowest corner set that still covers what it claims, for one metric."""
    safe = [r for r in results if r.metric == metric and r.safe]
    return min(safe, key=lambda r: r.waste) if safe else None
