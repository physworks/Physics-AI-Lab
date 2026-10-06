"""What the corners are supposed to bound.

Stage 2 has no circuit solver, so these are quantities computable from the
compact model alone.  They stand in for the circuit numbers a designer
actually cares about, and stage 3 replaces them with a real solve:

    ion       drive current -- sets how fast a stage can switch
    ioff      off-state current -- sets static leakage
    delay     a switching-delay proxy, C*V/Ion, so it rises as drive falls
    ion_ioff  the ratio, which trades the two against each other

A proxy is a proxy.  ``delay`` here ignores the input slope, the load's bias
dependence, and every parasitic.  The point of stage 3 is to find out whether
corners built against these still bound the real thing -- and if they do not,
that is the finding, not a failure to be tidied away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from .device import DeviceParams, drain_current

#: Operating points the proxies are evaluated at, volts.
VDD = 1.2
VG_ON, VD_ON = 1.2, 1.2
VG_OFF, VD_OFF = 0.0, 1.2

#: Load capacitance for the delay proxy, farads.  Only the scale depends on
#: it, and every method is compared at the same scale.
C_LOAD = 1.0e-15


def ion(p: DeviceParams) -> float:
    return float(drain_current(p, np.array([VG_ON]), np.array([VD_ON]))[0])


def ioff(p: DeviceParams) -> float:
    return float(drain_current(p, np.array([VG_OFF]), np.array([VD_OFF]))[0])


def delay(p: DeviceParams) -> float:
    """C * V / Ion -- larger is worse, and it is what a slow corner means."""
    return float(C_LOAD * VDD / max(ion(p), 1e-30))


def ion_ioff(p: DeviceParams) -> float:
    return float(ion(p) / max(ioff(p), 1e-30))


METRICS: Dict[str, Callable[[DeviceParams], float]] = {
    "ion": ion, "ioff": ioff, "delay": delay, "ion_ioff": ion_ioff,
}

#: Metrics whose spread is better read in log space, because they run over
#: orders of magnitude rather than percent.
LOG_METRICS = {"ioff", "ion_ioff"}


#: Both operating points in one array, so a device costs one model call.
_VG = np.array([VG_ON, VG_OFF])
_VD = np.array([VD_ON, VD_OFF])


def evaluate_all(devices) -> Dict[str, np.ndarray]:
    """Every metric for every device, one model evaluation each.

    ``drain_current`` carries a fixed per-call cost -- it solves the series
    resistance self-consistently -- that dwarfs the cost of the extra bias
    point.  Asking for both points together and deriving all four metrics from
    them turns six calls per device into one, which is the difference between
    this stage running in seconds and in minutes.
    """
    n = len(devices)
    on = np.empty(n)
    off = np.empty(n)
    for i, d in enumerate(devices):
        both = drain_current(d, _VG, _VD)
        on[i], off[i] = float(both[0]), float(both[1])

    on = np.maximum(on, 1e-30)
    off = np.maximum(off, 1e-30)
    return {"ion": on, "ioff": off,
            "delay": C_LOAD * VDD / on, "ion_ioff": on / off}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: Callable[[DeviceParams], float]
    log_scale: bool

    def evaluate(self, devices) -> np.ndarray:
        return evaluate_all(devices)[self.name]


def spec(name: str) -> MetricSpec:
    if name not in METRICS:
        raise KeyError(f"unknown metric {name!r}; have {sorted(METRICS)}")
    return MetricSpec(name, METRICS[name], name in LOG_METRICS)


def all_specs() -> Dict[str, MetricSpec]:
    return {n: spec(n) for n in METRICS}
