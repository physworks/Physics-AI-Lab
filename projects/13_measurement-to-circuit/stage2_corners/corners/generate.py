"""Four ways to turn a covariance into corners, and what each one assumes.

A corner model is a small set of parameter vectors that a circuit designer
simulates instead of the whole population.  The question every method answers
differently is *which directions in parameter space are worth a corner*.

    independent_box     every parameter at +-k sigma in every combination.
                        Classic fast/slow practice; assumes the parameters
                        vary independently.  They do not -- they share
                        physical causes (see physical.py) -- so the box
                        reaches well outside the k-sigma ellipsoid.
    one_at_a_time       each parameter to +-k sigma alone.  A sensitivity
                        sweep, and it fails the other way: it under-covers.
    pca_corners         +-k sigma along the eigenvectors of the covariance, so
                        the corners point along the directions the process
                        actually moves in.
    statistical_mc      draw from the fitted distribution and keep the sample.
                        No corners at all; the honest baseline for coverage.
    worst_case_distance for one metric, the point on the k-sigma ellipsoid
                        that worst-cases it.  Needs a metric up front, and
                        gives the tightest bound when one exists.

All of them work in log-parameter space, matching stage 1, so a corner is a
multiplicative deviation from nominal and every parameter stays positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np

from .device import PARAM_NAMES, DeviceParams


@dataclass
class CornerSet:
    method: str
    k: float
    log_offsets: np.ndarray             # (n_corners, n_params)
    nominal: Dict[str, float]
    labels: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.log_offsets.shape[0])

    def params(self) -> np.ndarray:
        nom = np.array([self.nominal[n] for n in PARAM_NAMES])
        return nom * np.exp(self.log_offsets)

    def as_devices(self) -> List[DeviceParams]:
        return [DeviceParams(**dict(zip(PARAM_NAMES, row)))
                for row in self.params()]

    def summary(self) -> Dict:
        return {"method": self.method, "k": self.k, "n_corners": len(self),
                "labels": list(self.labels), "notes": list(self.notes)}


def _sigma(cov: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def independent_box(cov: np.ndarray, nominal: Dict[str, float],
                    k: float = 3.0) -> CornerSet:
    """The hyperrectangle: every parameter at +-k sigma, in every combination.

    This is what a corner set looks like when the covariance has not been
    looked at -- the fast/slow corners of classic practice, built by taking
    each parameter to its own extreme and assuming the combination is as
    likely as the individual moves.  It is not.  With ``n`` independent
    parameters the box corner sits ``k * sqrt(n)`` away in Mahalanobis
    distance, so a "3 sigma" box corner is nearer 8 sigma for seven
    parameters, and correlation pushes it further still.

    It therefore over-covers, and the margin it spends doing so is the number
    this stage exists to put a value on.  All ``2^n`` corners are enumerated
    because a metric need not be monotonic in every parameter, so which corner
    is extreme is not known in advance.

    There is a second, sharper objection to it here.  The parameters are
    driven by six independent physical causes, so their covariance has rank
    six, not seven: one direction in parameter space is one the process cannot
    move in.  A box treats all seven as free, which means some of its corners
    are not merely unlikely but unreachable.
    """
    sd = _sigma(cov)
    n = len(PARAM_NAMES)
    offs, labels = [], []
    for mask in range(2 ** n):
        signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
        offs.append(signs * k * sd)
        labels.append("".join("+" if s > 0 else "-" for s in signs))
    return CornerSet("independent_box", k, np.array(offs), nominal, labels,
                     [f"all {2 ** n} corners of the +-{k} sigma box; "
                      "ignores correlation"])


def one_at_a_time(cov: np.ndarray, nominal: Dict[str, float],
                  k: float = 3.0) -> CornerSet:
    """Each parameter to +-k sigma on its own, the rest at nominal.

    A sensitivity sweep rather than a bounding corner set.  It is included
    because it is also common practice and because it fails in the opposite
    direction to the box: moving one parameter while the others sit at nominal
    cannot reproduce the swing of a population that moves all of them at once,
    so it *under*-covers.
    """
    sd = _sigma(cov)
    offs, labels = [], []
    for i, nm in enumerate(PARAM_NAMES):
        for sign, tag in ((+1.0, "hi"), (-1.0, "lo")):
            v = np.zeros(len(PARAM_NAMES))
            v[i] = sign * k * sd[i]
            offs.append(v)
            labels.append(f"{nm}_{tag}")
    return CornerSet("one_at_a_time", k, np.array(offs), nominal, labels,
                     ["one parameter at a time; a sensitivity sweep"])


def pca_corners(cov: np.ndarray, nominal: Dict[str, float], k: float = 3.0,
                n_axes: int | None = None, rel_tol: float = 1e-6) -> CornerSet:
    """+-k sigma along the principal axes of the covariance.

    Each corner is a *combination* of parameter shifts in the proportion the
    process actually produces, so it sits on the k-sigma ellipsoid rather than
    outside it.

    Axes carrying negligible variance are dropped.  That is not a numerical
    nicety: when the parameters are driven by fewer independent physical
    causes than there are parameters, the covariance is **rank deficient** and
    the surplus axes are directions the process cannot move in at all.
    Generating corners along them would add simulations that differ from
    nominal only by numerical noise.
    """
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]

    keep = int(np.sum(w > rel_tol * max(w[0], 1e-300)))
    keep = max(keep, 1) if n_axes is None else min(n_axes, max(keep, 1))

    offs, labels = [], []
    for a in range(keep):
        step = k * np.sqrt(max(w[a], 0.0)) * V[:, a]
        offs += [step, -step]
        labels += [f"pc{a + 1}_hi", f"pc{a + 1}_lo"]

    total = float(np.sum(np.clip(w, 0, None))) or 1.0
    frac = float(np.sum(w[:keep]) / total)
    notes = [f"{keep} principal axes covering {frac * 100:.1f}% of the "
             f"variance"]
    if keep < len(w):
        notes.append(f"{len(w) - keep} axis/axes dropped: the covariance has "
                     f"rank {keep} of {len(w)}, so the process cannot move "
                     f"in those directions")
    return CornerSet("pca_corners", k, np.array(offs), nominal, labels, notes)


def statistical_mc(cov: np.ndarray, nominal: Dict[str, float],
                   n: int = 500, seed: int = 0, k: float = 3.0) -> CornerSet:
    """A sample from the fitted distribution; not corners, the comparison."""
    rng = np.random.default_rng(seed)
    w, V = np.linalg.eigh(cov)
    A = V @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
    offs = rng.standard_normal((n, cov.shape[0])) @ A.T
    return CornerSet("statistical_mc", k, offs, nominal,
                     [f"mc{i}" for i in range(n)],
                     [f"{n} draws from the fitted covariance"])


def worst_case_distance(cov: np.ndarray, nominal: Dict[str, float],
                        metric: Callable[[DeviceParams], float],
                        k: float = 3.0, refine: int = 2) -> CornerSet:
    """The points on the k-sigma ellipsoid that minimise and maximise a metric.

    To first order a smooth metric is extremised on the ellipsoid at

        x = +- k * Sigma g / sqrt(g^T Sigma g)

    with ``g`` the gradient of the metric in log-parameter space.  The
    gradient is re-evaluated at the candidate point a couple of times, which
    is enough when the metric is mildly nonlinear and is what makes this a
    bound rather than a guess.

    It needs the metric in advance, so it answers a narrower question than the
    other methods: not "what can the process do" but "what can it do to *this*
    number".
    """
    nom = np.array([nominal[n] for n in PARAM_NAMES])

    def grad(x: np.ndarray, step: float = 1e-4) -> np.ndarray:
        g = np.zeros_like(x)
        for i in range(x.size):
            hi, lo = x.copy(), x.copy()
            hi[i] += step
            lo[i] -= step
            fh = metric(DeviceParams(**dict(zip(PARAM_NAMES,
                                                nom * np.exp(hi)))))
            fl = metric(DeviceParams(**dict(zip(PARAM_NAMES,
                                                nom * np.exp(lo)))))
            g[i] = (fh - fl) / (2.0 * step)
        return g

    offs, labels = [], []
    for sign, tag in ((+1.0, "max"), (-1.0, "min")):
        x = np.zeros(len(PARAM_NAMES))
        for _ in range(max(refine, 1)):
            g = grad(x)
            denom = float(np.sqrt(max(g @ cov @ g, 1e-300)))
            x = sign * k * (cov @ g) / denom
        offs.append(x)
        labels.append(f"wcd_{tag}")

    return CornerSet("worst_case_distance", k, np.array(offs), nominal, labels,
                     ["extremes of one metric on the k-sigma ellipsoid"])


def all_methods(cov: np.ndarray, nominal: Dict[str, float], k: float = 3.0,
                metric: Callable[[DeviceParams], float] | None = None,
                mc_n: int = 500, seed: int = 0) -> Dict[str, CornerSet]:
    out = {"independent_box": independent_box(cov, nominal, k),
           "one_at_a_time": one_at_a_time(cov, nominal, k),
           "pca_corners": pca_corners(cov, nominal, k),
           "statistical_mc": statistical_mc(cov, nominal, mc_n, seed, k)}
    if metric is not None:
        out["worst_case_distance"] = worst_case_distance(cov, nominal,
                                                         metric, k)
    return out
