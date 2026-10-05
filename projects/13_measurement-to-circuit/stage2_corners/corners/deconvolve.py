"""Separating process spread from the noise in measuring it.

What an extraction produces is not the process distribution.  Every device is
measured with finite precision, so the parameters that come back are the true
ones plus an estimation error:

    observed  =  process  (+)  extraction

In log-parameter space, and treating the two as independent, the covariances
add.  Build corners from the observed spread and they are too wide by whatever
the measurement contributed -- margin spent on the instrument rather than on
the devices.

The extraction term is exactly the quantity stage 1 minimised, and stage 1's
Fisher analysis predicts it without any Monte Carlo.  So the two stages meet
here: a better measurement design does not only give better parameters, it
makes the corners built from them tighter.

Subtracting covariances can produce a matrix that is not positive
semi-definite when the extraction term is comparable to the process term, so
the result is projected back onto the PSD cone and the size of that projection
is reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .device import (DeviceParams, DesignConstraints, NoiseModel, PARAM_NAMES,
                     build_pool, default_required_regions, extract_subset,
                     full_objective, greedy_design, information, measure)
from .physical import Population


@dataclass
class Deconvolution:
    observed: np.ndarray
    extraction: np.ndarray
    process: np.ndarray                 # observed - extraction, projected PSD
    psd_correction: float               # how much the projection moved it
    names: List[str]

    def inflation(self) -> Dict[str, float]:
        """Per parameter, how much wider the observed spread is than the process."""
        o = np.sqrt(np.clip(np.diag(self.observed), 0, None))
        p = np.sqrt(np.clip(np.diag(self.process), 0, None))
        return {n: float(o[i] / p[i]) if p[i] > 0 else float("inf")
                for i, n in enumerate(self.names)}

    def summary(self) -> Dict:
        return {"psd_correction": self.psd_correction,
                "inflation": self.inflation(),
                "observed_sigma": {
                    n: float(np.sqrt(max(self.observed[i, i], 0.0)))
                    for i, n in enumerate(self.names)},
                "process_sigma": {
                    n: float(np.sqrt(max(self.process[i, i], 0.0)))
                    for i, n in enumerate(self.names)}}


def project_psd(A: np.ndarray) -> tuple[np.ndarray, float]:
    """Nearest positive semi-definite matrix, and how far it had to move."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    clipped = np.clip(w, 0.0, None)
    moved = float(np.sum(np.abs(w - clipped)) / max(np.sum(np.abs(w)), 1e-300))
    return (V * clipped) @ V.T, moved


def extraction_covariance(nominal: DeviceParams, vgs, vds,
                          noise: NoiseModel,
                          params: Sequence[str] | None = None) -> np.ndarray:
    """Stage 1's predicted estimation covariance for one measurement design.

    This is the inverse Fisher matrix in log-parameter space -- the same object
    whose diagonal stage 1 reported as a relative 1-sigma, used here whole
    because the off-diagonals matter for corners.
    """
    params = list(params or PARAM_NAMES)
    res = information(nominal, vgs, vds, noise, params)
    n = res.matrix.shape[0]
    ridge = 1e-10 * max(float(np.trace(res.matrix)) / max(n, 1), 1.0)
    return np.linalg.inv(res.matrix + ridge * np.eye(n))


def design_from_stage1(nominal: DeviceParams, noise: NoiseModel,
                       n_points: int = 12, max_distinct_vds: int = 3):
    """The measurement plan stage 1 selects, reused unchanged."""
    cons = DesignConstraints(n_points=n_points,
                             max_distinct_vds=max_distinct_vds,
                             required_regions=default_required_regions())
    pool = build_pool(nominal, cons)
    design = greedy_design(nominal, pool, cons, noise,
                           full_objective(nominal.to_dict(), "D"),
                           arm="stage1")
    return design


def measure_population(pop: Population, design, noise: NoiseModel,
                       seed: int = 0) -> tuple[np.ndarray, int]:
    """Extract every device from the designed bias points.

    Returns the estimated parameters and the number of devices whose
    extraction did not complete.  Failures are dropped rather than retried;
    a corner model built from the ones that converged is the realistic case.
    """
    rows, failures = [], 0
    for i, dev in enumerate(pop.as_devices()):
        ids = measure(dev, design.vgs, design.vds,
                      noise_rel=noise.sigma_rel, noise_floor=noise.i_floor,
                      seed=seed + i)
        try:
            est = extract_subset(design.vgs, design.vds, ids, PARAM_NAMES)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            failures += 1
            continue
        rows.append([getattr(est, n) for n in PARAM_NAMES])
    return np.array(rows, dtype=float), failures


def deconvolve(observed_params: np.ndarray, nominal: Dict[str, float],
               extraction: np.ndarray) -> Deconvolution:
    """Observed covariance minus the extraction covariance."""
    nom = np.array([nominal[n] for n in PARAM_NAMES])
    obs = np.cov(np.log(observed_params / nom), rowvar=False)
    process, moved = project_psd(obs - extraction)
    return Deconvolution(observed=obs, extraction=extraction,
                         process=process, psd_correction=moved,
                         names=list(PARAM_NAMES))
