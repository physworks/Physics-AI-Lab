"""A compact MOSFET model, written to be *extracted from* rather than to be exact.

Scope, stated up front: this is NOT BSIM-CMG or any industry-standard compact
model.  It is a deliberately small analytic model with seven parameters, chosen
so that the extraction procedure and the identifiability analysis are the
subject of this project rather than the model itself.

The core is the EKV-style interpolation function

    F(u) = ln^2(1 + exp(u/2)),   Id = Ispec * [F(up) - F(up - ud)]

which reduces to an exponential in weak inversion and to the usual quadratic in
strong inversion, with continuous derivatives in between.  Using a single
expression across both regimes matters here: a kink at threshold would make the
staged extraction ill-posed exactly where two stages hand off to each other.

The seven parameters and the regime that determines each:

    vth0    threshold voltage              subthreshold intercept, linear region
    ss      subthreshold swing [V/dec]     slope of log(Id) below threshold
    mu0     low-field mobility             linear region slope
    theta   mobility degradation           linear region curvature at high Vgs
    rs      series resistance              linear region roll-off at high Vgs
    eta     DIBL coefficient               Vth shift between low and high Vds
    vsat    saturation velocity            saturation current level

Each parameter dominates in a different measurement regime.  That is what makes
staged extraction possible -- and, where two of them act on the same regime,
what makes the identifiability analysis necessary.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Tuple

import numpy as np

# Constants and fixed geometry.  These are NOT extracted: they are known from
# the test structure, as they would be in practice.
PHI_T = 0.02585           # kT/q at 300 K [V]
COX = 1.2e-2              # gate oxide capacitance per unit area [F/m^2]
W_OVER_L = 10.0           # channel width / length
L_CH = 1.0e-6             # channel length [m]
LN10 = np.log(10.0)


@dataclass
class DeviceParams:
    """The parameter vector an extraction is trying to recover."""

    vth0: float = 0.40        # V
    ss: float = 0.075         # V/decade
    mu0: float = 0.030        # m^2/Vs
    theta: float = 0.35       # 1/V
    rs: float = 180.0         # ohm (source + drain)
    eta: float = 0.045        # V/V (DIBL)
    vsat: float = 8.0e4       # m/s

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=float)

    @classmethod
    def from_array(cls, x) -> "DeviceParams":
        return cls(**{f.name: float(v) for f, v in zip(fields(cls), np.asarray(x))})

    @staticmethod
    def names() -> List[str]:
        return [f.name for f in fields(DeviceParams)]

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    def copy_with(self, **kw) -> "DeviceParams":
        d = self.to_dict()
        d.update({k: float(v) for k, v in kw.items()})
        return DeviceParams(**d)


# Physically meaningful bounds.  A value outside these is not a fit needing more
# iterations -- it is a result that must not be accepted.
BOUNDS: Dict[str, Tuple[float, float]] = {
    "vth0": (0.05, 1.20),
    "ss": (0.0595, 0.200),     # ~59.5 mV/dec is the 300 K thermal floor
    "mu0": (0.002, 0.090),
    "theta": (0.0, 2.0),
    "rs": (0.0, 2000.0),
    "eta": (0.0, 0.30),
    "vsat": (2.0e4, 2.0e5),
}

# Which stage of the extraction determines each parameter.  Used by the report
# and by the identifiability analysis.
STAGE_OF = {
    "ss": "subthreshold", "vth0": "subthreshold",
    "eta": "dibl",
    "mu0": "linear", "theta": "linear", "rs": "linear",
    "vsat": "saturation",
}


def _F(u: np.ndarray) -> np.ndarray:
    """EKV interpolation function, evaluated without overflow."""
    h = 0.5 * np.asarray(u, dtype=float)
    # log1p(exp(h)) == h + log1p(exp(-h)) for large h
    big = h > 30.0
    out = np.where(big, h, np.log1p(np.exp(np.clip(h, -60.0, 30.0))))
    return out * out


def _current_internal(p: DeviceParams, vgs: np.ndarray,
                      vds_int: np.ndarray) -> np.ndarray:
    """Id given the voltage that actually appears across the intrinsic device."""
    n = p.ss / (LN10 * PHI_T)                      # subthreshold ideality
    vth = p.vth0 - p.eta * vds_int
    vp = (vgs - vth) / n                           # pinch-off voltage

    # Smoothed overdrive, used only for field-dependent mobility.
    vov = n * PHI_T * np.log1p(np.exp(np.clip(vp / (n * PHI_T), -60.0, 60.0)))

    mu_vert = p.mu0 / (1.0 + p.theta * vov)        # vertical-field degradation
    esat = 2.0 * p.vsat / np.maximum(mu_vert, 1e-9)
    mu_eff = mu_vert / (1.0 + vds_int / np.maximum(esat * L_CH, 1e-12))

    ispec = 2.0 * n * mu_eff * COX * W_OVER_L * PHI_T * PHI_T
    up = vp / PHI_T
    ud = vds_int / PHI_T
    return ispec * (_F(up) - _F(up - ud))


def drain_current(p: DeviceParams, vgs, vds) -> np.ndarray:
    """Id(Vgs, Vds) including series resistance, solved self-consistently.

    Series resistance is applied by damped fixed-point iteration on
    Vds_internal = Vds - Id * rs.  Damping keeps it stable at high Vgs, where
    the loop gain approaches one -- which is also where rs and mu0 become hard
    to tell apart.
    """
    vgs = np.asarray(vgs, dtype=float)
    vds = np.asarray(vds, dtype=float)

    vds_int = vds.copy()
    for _ in range(40):
        ids = _current_internal(p, vgs, vds_int)
        target = np.maximum(vds - ids * p.rs, 1e-6)
        new = 0.5 * vds_int + 0.5 * target          # damping factor 0.5
        if np.max(np.abs(new - vds_int)) < 1e-12:
            vds_int = new
            break
        vds_int = new

    return np.maximum(_current_internal(p, vgs, vds_int), 1e-15)


def sweep_grid(vgs_range=(0.0, 1.2), n_vgs=61,
               vds_list=(0.05, 0.6, 1.2)) -> Tuple[np.ndarray, np.ndarray]:
    """Standard Id-Vg sweeps at several Vds.

    Low Vds isolates the linear region (mobility, series R); high Vds gives
    saturation; the pair gives the DIBL shift.
    """
    vgs = np.linspace(*vgs_range, n_vgs)
    vg, vd = np.meshgrid(vgs, np.asarray(vds_list, dtype=float), indexing="ij")
    return vg.ravel(), vd.ravel()


def measure(p: DeviceParams, vgs, vds, noise_rel: float = 0.01,
            noise_floor: float = 1e-12, seed: int = 0) -> np.ndarray:
    """Synthetic measurement: true current plus instrument noise.

    Two noise terms, because they dominate in different regimes and that
    asymmetry is what limits which parameters can be recovered:
      * relative noise -- strong inversion (large currents)
      * noise floor    -- deep subthreshold (small currents)
    """
    rng = np.random.default_rng(seed)
    ideal = drain_current(p, vgs, vds)
    noisy = ideal * (1.0 + noise_rel * rng.standard_normal(ideal.shape))
    noisy = noisy + noise_floor * rng.standard_normal(ideal.shape)
    return np.maximum(noisy, 0.1 * noise_floor)
