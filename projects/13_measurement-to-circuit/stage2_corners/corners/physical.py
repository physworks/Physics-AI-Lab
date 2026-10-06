"""Where parameter correlation actually comes from.

A covariance matrix written down by hand encodes its own answer.  This module
instead samples *physical* quantities that vary across a wafer -- oxide
thickness, channel dimensions, interface traps, contact resistance, work
function -- and derives what each one does to the compact-model parameters.
The parameter correlation is then a consequence of shared physical causes
rather than a stipulation.

One subtlety drives the whole map.  The model in project 12 holds ``COX``,
``W_OVER_L`` and ``L_CH`` as module constants, so a device whose oxide is
thinner than nominal cannot express that through ``COX``.  The extraction has
no choice but to absorb it into the seven fitted parameters: a thinner oxide
raises the real ``Cox``, and since drain current goes as ``mu * Cox``, the
*fitted* ``mu0`` rises to compensate.  That is not a modelling error, it is
what fitting a fixed-geometry model to a varying device does, and it is the
origin of the strongest parameter correlations here.

The sensitivities below are first-order coefficients from textbook device
relations, not values fitted to data.  Each is annotated with the relation it
comes from so a reader can disagree with a specific number rather than with
the whole table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .device import PARAM_NAMES, DeviceParams

#: Thermal floor of the subthreshold swing at 300 K, V/decade.
SS_FLOOR = 0.0595


@dataclass(frozen=True)
class PhysicalVariable:
    """One physical cause of device-to-device variation."""

    name: str
    rel_sigma: float
    description: str


#: Independent per-parameter variation, on top of the shared causes.
#:
#: Without it, two parameters driven by the same single cause come out
#: perfectly correlated -- the first version of this module produced an
#: eta/vsat correlation of exactly -1.00, which no real device population
#: shows.  Local effects (random dopant fluctuation, line-edge roughness,
#: local strain) act on each parameter separately and break that degeneracy.
#: They are small relative to the shared causes, so the covariance stays
#: strongly anisotropic: the point that the parameters do not fill seven
#: independent directions survives, as a condition number rather than as an
#: exact rank deficiency.
LOCAL_SIGMA: Dict[str, float] = {
    "vth0": 0.006, "ss": 0.004, "mu0": 0.008, "theta": 0.010,
    "rs": 0.015, "eta": 0.020, "vsat": 0.008,
}


def default_variables() -> List[PhysicalVariable]:
    """Plausible wafer-scale spreads.  Magnitudes are illustrative."""
    return [
        PhysicalVariable("tox", 0.020,
                         "gate oxide thickness; sets Cox"),
        PhysicalVariable("L", 0.030,
                         "channel length; sets drive current and DIBL"),
        PhysicalVariable("W", 0.020,
                         "channel width; sets drive current"),
        PhysicalVariable("dit", 0.100,
                         "interface trap density; sets subthreshold swing"),
        PhysicalVariable("rc", 0.080,
                         "contact and extension resistance"),
        PhysicalVariable("vfb", 0.015,
                         "flatband / work function, as a fraction of vth0"),
    ]


def _ss_split(ss_nominal: float = 0.075) -> tuple[float, float]:
    """Split the subthreshold swing into its floor and its capacitive excess.

    ``SS = (ln10 kT/q) * (1 + (Cd + Cit)/Cox)``.  The excess over the thermal
    floor is the ``(Cd + Cit)/Cox`` term, and only that part responds to oxide
    thickness or trap density.
    """
    excess = ss_nominal / SS_FLOOR - 1.0        # (Cd + Cit) / Cox
    return excess, excess / (1.0 + excess)      # excess, its share of SS


_EXCESS, _EXCESS_SHARE = _ss_split()
_CIT_FRACTION = 0.5          # share of (Cd + Cit) attributed to traps


#: d(ln parameter) / d(ln physical variable), first order.
#:
#: tox   Cox ~ 1/tox.
#:       vth0  depletion-charge term ~ 1/Cox ~ tox; it is roughly a third of
#:             vth0 at this nominal, so +0.35.
#:       ss    only the capacitive excess responds, hence the share above.
#:       mu0   COX is fixed in the model, so the fitted mu0 absorbs the real
#:             Cox change: Id ~ mu*Cox gives -1.0.
#:       theta mobility degradation follows the vertical field, ~1/tox.
#: L     mu0 absorbs W/L for the same reason (-1.0); DIBL rises steeply as L
#:       falls; mild Vth roll-off.
#: dit   raises the trap part of the capacitive excess, and shifts vth0 a
#:       little through trapped charge.
#: rc    series resistance, one for one.
#: vfb   a work-function shift lands directly on vth0.
SENSITIVITY: Dict[str, Dict[str, float]] = {
    "tox": {"vth0": +0.35, "ss": +_EXCESS_SHARE, "mu0": -1.00,
            "theta": -1.00},
    "L": {"mu0": -1.00, "eta": -3.00, "vth0": +0.15, "vsat": +0.20},
    "W": {"mu0": +1.00},
    "dit": {"ss": +_EXCESS_SHARE * _CIT_FRACTION, "vth0": +0.05},
    "rc": {"rs": +1.00},
    "vfb": {"vth0": +1.00},
}


@dataclass
class Population:
    """A sampled set of devices, with the physical draws kept alongside."""

    params: np.ndarray                      # (n_devices, n_params)
    names: List[str]
    physical: Dict[str, np.ndarray] = field(default_factory=dict)
    nominal: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.params.shape[0])

    def as_devices(self) -> List[DeviceParams]:
        return [DeviceParams(**dict(zip(self.names, row)))
                for row in self.params]

    def log_deviation(self) -> np.ndarray:
        """ln(p / p_nominal), the space the sensitivities are written in."""
        nom = np.array([self.nominal[n] for n in self.names])
        return np.log(self.params / nom)

    def covariance(self) -> np.ndarray:
        """Covariance in log-parameter space, matching stage 1's convention."""
        return np.cov(self.log_deviation(), rowvar=False)

    def correlation(self) -> np.ndarray:
        c = self.covariance()
        sd = np.sqrt(np.clip(np.diag(c), 1e-300, None))
        out = c / np.outer(sd, sd)
        return np.nan_to_num(out, nan=0.0)


def process_covariance(variables: Sequence[PhysicalVariable] | None = None,
                       names: Sequence[str] | None = None,
                       local: bool = True) -> np.ndarray:
    """Analytic log-space covariance implied by the physical causes.

    With independent physical draws, ``Sigma = S diag(sigma_x^2) S^T + L``
    where ``S`` is the sensitivity matrix and ``L`` the diagonal of local,
    per-parameter variation.  Computing it directly as well as by sampling
    gives the tests something to check the sampler against.
    """
    variables = list(variables or default_variables())
    names = list(names or PARAM_NAMES)
    S = np.zeros((len(names), len(variables)))
    for j, var in enumerate(variables):
        for p, s in SENSITIVITY.get(var.name, {}).items():
            if p in names:
                S[names.index(p), j] = s
    d = np.diag([v.rel_sigma ** 2 for v in variables])
    cov = S @ d @ S.T
    if local:
        cov = cov + np.diag([LOCAL_SIGMA.get(n, 0.0) ** 2 for n in names])
    return cov


def anisotropy(cov: np.ndarray | None = None) -> Dict[str, float]:
    """How far the covariance is from treating the parameters as independent."""
    cov = process_covariance() if cov is None else cov
    w = np.sort(np.linalg.eigvalsh(cov))[::-1]
    w = np.clip(w, 1e-300, None)
    return {"condition_number": float(w[0] / w[-1]),
            "axes_for_90pct_variance": int(
                np.searchsorted(np.cumsum(w) / np.sum(w), 0.90) + 1),
            "n_params": len(w)}


def sample_population(n_devices: int = 500,
                      variables: Sequence[PhysicalVariable] | None = None,
                      nominal: DeviceParams | None = None,
                      seed: int = 0,
                      clip_sigma: float = 4.0,
                      local: bool = True) -> Population:
    """Draw devices by sampling physics, not by sampling parameters."""
    variables = list(variables or default_variables())
    nominal = nominal or DeviceParams()
    nom = nominal.to_dict()
    rng = np.random.default_rng(seed)

    draws = {}
    for var in variables:
        z = rng.standard_normal(n_devices)
        draws[var.name] = np.clip(z, -clip_sigma, clip_sigma) * var.rel_sigma

    log_dev = np.zeros((n_devices, len(PARAM_NAMES)))
    for var in variables:
        for p, s in SENSITIVITY.get(var.name, {}).items():
            log_dev[:, PARAM_NAMES.index(p)] += s * draws[var.name]

    if local:
        for p, sg in LOCAL_SIGMA.items():
            z = np.clip(rng.standard_normal(n_devices), -clip_sigma,
                        clip_sigma)
            log_dev[:, PARAM_NAMES.index(p)] += sg * z

    params = np.array([nom[n] for n in PARAM_NAMES]) * np.exp(log_dev)

    # The thermal floor is physics, not a modelling choice: a sampled device
    # may not sit below it.
    params[:, PARAM_NAMES.index("ss")] = np.maximum(
        params[:, PARAM_NAMES.index("ss")], SS_FLOOR * 1.001)

    return Population(params=params, names=list(PARAM_NAMES),
                      physical=draws, nominal=nom)


def dominant_cause(param: str) -> str:
    """Which physical variable contributes most variance to one parameter."""
    best, best_v = "local", LOCAL_SIGMA.get(param, 0.0) ** 2
    for var in default_variables():
        s = SENSITIVITY.get(var.name, {}).get(param, 0.0)
        v = (s * var.rel_sigma) ** 2
        if v > best_v:
            best, best_v = var.name, v
    return best
