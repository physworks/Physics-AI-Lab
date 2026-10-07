"""The circuits, and the metrics stage 2's corners now have to bound.

All NMOS-with-load, because the model is an NMOS model.  Each builder returns
a circuit plus the nodes a measurement reads, so the metric functions below do
not have to know how the netlist was wired.

The metrics replace stage 2's proxies:

    stage 2 proxy          stage 3 measurement
    ion                ->  drive current at the inverter's low output
    delay = C*V/Ion    ->  propagation delay from an actual transient
    (none)             ->  switching threshold and small-signal gain
    (none)             ->  current-mirror ratio error against output voltage

``delay`` is the one to watch.  Stage 2 built its tightest corner set against
``C*V/Ion``, which depends on the drive current at a single bias.  A real
propagation delay depends on the whole I-V trajectory the output swings
through, so a corner chosen for the proxy need not bound the real thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .device import DeviceParams
from .netlist import Capacitor, Circuit, Nmos, Resistor, VoltageSource
from .solver import solve_dc, solve_transient

VDD = 1.2
R_LOAD = 2.0e3          # ohm; sets the inverter's low level with this device
C_LOAD = 20.0e-15       # farad


def inverter(p: DeviceParams, vin: float, r_load: float = R_LOAD,
             vdd: float = VDD) -> Circuit:
    """Resistive-load inverter: VDD - R - out - NMOS - ground."""
    c = Circuit()
    c.add(VoltageSource("vdd", "0", vdd, name="VDD"))
    c.add(VoltageSource("in", "0", vin, name="VIN"))
    c.add(Resistor("vdd", "out", r_load, name="RL"))
    c.add(Nmos("out", "in", "0", p, name="M1"))
    return c


def current_mirror(p: DeviceParams, i_ref: float, v_out: float,
                   vdd: float = VDD) -> Circuit:
    """Diode-connected reference device mirroring into a forced output.

    The output is held by a voltage source so the mirror current can be read
    directly as that source's current, which is also what makes the mirror a
    clean probe of output conductance: sweeping ``v_out`` traces exactly the
    region where ``gds`` goes negative.
    """
    c = Circuit()
    c.add(VoltageSource("vdd", "0", vdd, name="VDD"))
    c.add(VoltageSource("out", "0", v_out, name="VOUT"))
    r_bias = max((vdd - 0.6) / max(i_ref, 1e-9), 1.0)
    c.add(Resistor("vdd", "ref", r_bias, name="RB"))
    c.add(Nmos("ref", "ref", "0", p, name="M1"))
    c.add(Nmos("out", "ref", "0", p, name="M2"))
    return c


def source_follower(p: DeviceParams, vin: float, r_load: float = R_LOAD,
                    vdd: float = VDD) -> Circuit:
    c = Circuit()
    c.add(VoltageSource("vdd", "0", vdd, name="VDD"))
    c.add(VoltageSource("in", "0", vin, name="VIN"))
    c.add(Nmos("vdd", "in", "out", p, name="M1"))
    c.add(Resistor("out", "0", r_load, name="RL"))
    return c


def ring_oscillator(p: DeviceParams, stages: int = 3,
                    r_load: float = R_LOAD, c_load: float = C_LOAD,
                    vdd: float = VDD) -> Circuit:
    """An odd ring of resistive-load inverters, each loaded by a capacitor."""
    if stages % 2 == 0:
        raise ValueError("a ring oscillator needs an odd number of stages")
    c = Circuit()
    c.add(VoltageSource("vdd", "0", vdd, name="VDD"))
    names = [f"n{i}" for i in range(stages)]
    for i, node in enumerate(names):
        drive = names[i - 1]
        c.add(Resistor("vdd", node, r_load, name=f"RL{i}"))
        c.add(Nmos(node, drive, "0", p, name=f"M{i}"))
        c.add(Capacitor(node, "0", c_load, name=f"CL{i}"))
    return c


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

@dataclass
class TransferCurve:
    vin: np.ndarray
    vout: np.ndarray
    converged: np.ndarray

    def switching_threshold(self) -> float:
        """Input at which vout crosses VDD/2."""
        ok = self.converged
        vi, vo = self.vin[ok], self.vout[ok]
        mid = VDD / 2.0
        for i in range(vo.size - 1):
            if vo[i] >= mid > vo[i + 1]:
                f = (vo[i] - mid) / (vo[i] - vo[i + 1])
                return float(vi[i] + f * (vi[i + 1] - vi[i]))
        return float("nan")

    def peak_gain(self) -> float:
        ok = self.converged
        vi, vo = self.vin[ok], self.vout[ok]
        if vi.size < 3:
            return float("nan")
        return float(np.max(np.abs(np.gradient(vo, vi))))

    def v_low(self) -> float:
        ok = self.converged
        return float(np.min(self.vout[ok])) if ok.any() else float("nan")


def transfer_curve(p: DeviceParams, n: int = 41,
                   vin_range: Tuple[float, float] = (0.0, VDD)
                   ) -> Tuple[TransferCurve, List[Dict]]:
    """DC sweep of the inverter, carrying each solve's trace."""
    vins = np.linspace(vin_range[0], vin_range[1], n)
    vout, ok, traces = [], [], []
    x_prev = None
    for v in vins:
        c = inverter(p, v)
        r = solve_dc(c, x_prev)
        x_prev = r.x if r.converged else None       # restart on failure
        vout.append(c.voltage(r.x, "out"))
        ok.append(r.converged)
        traces.append({"vin": float(v), **r.trace()})
    return (TransferCurve(vins, np.array(vout), np.array(ok, dtype=bool)),
            traces)


def mirror_error(p: DeviceParams, i_ref: float = 2.0e-4,
                 v_outs: Tuple[float, ...] = (0.3, 0.6, 0.9, 1.1)
                 ) -> Dict[str, float]:
    """Mirror current against output voltage.

    An ideal mirror holds its output current flat as ``v_out`` rises. A device
    with negative output conductance does the opposite of the usual error: the
    current *falls* with increasing output voltage, which is a negative output
    resistance at the mirror's output node.
    """
    currents, traces = [], []
    for vo in v_outs:
        c = current_mirror(p, i_ref, vo)
        r = solve_dc(c)
        m2 = [d for d in c.devices() if d.name == "M2"][0]
        currents.append(m2.last.get("ids", float("nan"))
                        if r.converged else float("nan"))
        traces.append(r.converged)
    cur = np.array(currents)
    good = np.isfinite(cur)
    out = {"currents": cur.tolist(), "all_converged": bool(np.all(traces))}
    if good.sum() >= 2:
        ref = cur[good][0]
        out["spread_percent"] = float(
            (cur[good].max() - cur[good].min()) / abs(ref) * 100.0)
        slope = np.polyfit(np.array(v_outs)[good], cur[good], 1)[0]
        out["output_conductance"] = float(slope)
        out["negative_output_resistance"] = bool(slope < 0)
    return out


def propagation_delay(p: DeviceParams, stages: int = 3,
                      dt: float = 2.0e-12, t_stop: float = 4.0e-9
                      ) -> Dict[str, float]:
    """Stage delay from a ring oscillator's period, if it oscillates."""
    c = ring_oscillator(p, stages)
    r = solve_transient(c, t_stop, dt, initial_perturb=0.25,
                        perturb_node="n0")
    out = {"converged": r.converged,
           "failed_at": r.failed_at,
           "mean_newton_iterations": float(np.mean(r.newton_iterations))
           if r.newton_iterations else float("nan")}
    period = r.period_from("n0", skip=0.25 * t_stop)
    out["period"] = period
    out["oscillated"] = period is not None
    out["stage_delay"] = (period / (2.0 * stages)) if period else float("nan")
    out["frequency"] = (1.0 / period) if period else float("nan")
    return out
