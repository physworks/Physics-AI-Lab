"""Known faults, injected so the diagnosis has an answer key.

Job 1 and Job 2 produced almost no convergence failures -- the solver handles
this model well, and the negative output conductance it does have turned out
not to be reachable as a circuit failure.  Without failures there is nothing
to diagnose, so the failures are manufactured, each from a defect that happens
to real compact models.

The injected fault is the ground truth.  What the diagnoser sees is the
solver's trace: residual history, Jacobian conditioning, and what each device
was doing.  **No field in the trace names the fault.**  The trace carries
symptoms -- a transconductance that has gone to zero, an inner loop that did
not converge, a retry from a different start that succeeded -- and the
diagnosis is the inference from them.  A trace field called ``clamp_active``
would make the task arithmetic rather than judgement, so there is none.

The six classes:

``none``                      converged; nothing wrong
``c1_kink``                   a clamp flattens the current; the derivative
                              drops to zero and Newton has no direction left
``negative_gds``              output conductance driven negative hard enough
                              to flip the node conductance
``truncated_inner_loop``      the series-resistance loop is capped, so the
                              current and the Jacobian disagree
``bad_initial_guess``         no model defect; the start was simply too far
``singular_operating_point``  no solution exists at this bias
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .device import DeviceParams, Evaluation, evaluate
from .netlist import Circuit, CurrentSource, Nmos, Resistor, VoltageSource

FAULT_CLASSES = ("none", "c1_kink", "negative_gds", "truncated_inner_loop",
                 "bad_initial_guess", "singular_operating_point")


@dataclass
class FaultyNmos(Nmos):
    """An Nmos whose evaluation is perturbed in one named way."""

    fault: str = "none"
    strength: float = 1.0

    def stamp(self, x, F, J, nmap, extra_row):
        vd = x[nmap[self.d]] if nmap[self.d] >= 0 else 0.0
        vg = x[nmap[self.g]] if nmap[self.g] >= 0 else 0.0
        vs = x[nmap[self.s]] if nmap[self.s] >= 0 else 0.0
        vgs, vds = vg - vs, vd - vs

        ev = evaluate(self.params, np.array([vgs]), np.array([vds]))
        ids, gm, gds = float(ev.ids[0]), float(ev.gm[0]), float(ev.gds[0])
        inner_it, inner_ok = ev.iterations, ev.converged

        if self.fault == "c1_kink":
            # A step in the current at a drain-voltage threshold: the value
            # jumps while the derivative reports the smooth model, so Newton
            # is handed a direction that does not lead to the root it is
            # chasing and chatters across the boundary.
            #
            # A simple ceiling was tried first and did not fail at all -- a
            # clamped device is just a constant current source, and the node
            # equation still has one solution. A discontinuity in the value is
            # what actually breaks a solve; a discontinuity in the slope only
            # slows it.
            if vds > self.strength:
                ids *= 1.25

        elif self.fault == "negative_gds":
            gds = -abs(gds) * self.strength - 1e-3 * self.strength

        elif self.fault == "truncated_inner_loop":
            # Current from a deliberately short fixed-point run, derivatives
            # from the converged solve: the two no longer describe the same
            # device, which is what a capped inner loop really does.
            ids = _short_loop_current(self.params, vgs, vds, n=3)
            inner_it, inner_ok = 3, False

        self.last = {"ids": ids, "gm": gm, "gds": gds, "vgs": vgs, "vds": vds,
                     "inner_iterations": inner_it, "inner_converged": inner_ok}

        i = nmap[self.d]
        if i >= 0:
            F[i] += ids
            for node, val in ((self.d, gds), (self.g, gm),
                              (self.s, -(gm + gds))):
                k = nmap[node]
                if k >= 0:
                    J[i, k] += val
        i = nmap[self.s]
        if i >= 0:
            F[i] -= ids
            for node, val in ((self.d, -gds), (self.g, -gm),
                              (self.s, gm + gds)):
                k = nmap[node]
                if k >= 0:
                    J[i, k] += val


def _short_loop_current(p: DeviceParams, vgs: float, vds: float,
                        n: int = 3) -> float:
    """Project 12's damped fixed point, stopped early on purpose."""
    from cmext.model import _current_internal
    vg = np.array([vgs], dtype=float)
    vd = np.array([vds], dtype=float)
    vdi = vd.copy()
    for _ in range(n):
        ids = _current_internal(p, vg, vdi)
        target = np.maximum(vd - ids * p.rs, 1e-6)
        vdi = 0.5 * vdi + 0.5 * target
    return float(_current_internal(p, vg, vdi)[0])


@dataclass
class Case:
    """One circuit to solve, and the fault that was put into it."""

    name: str
    truth: str
    circuit: Circuit
    x0: np.ndarray | None = None
    note: str = ""


def _inverter_with(p: DeviceParams, vin: float, fault: str, strength: float,
                   r_load: float = 2.0e3, vdd: float = 1.2) -> Circuit:
    c = Circuit()
    c.add(VoltageSource("vdd", "0", vdd, name="VDD"))
    c.add(VoltageSource("in", "0", vin, name="VIN"))
    c.add(Resistor("vdd", "out", r_load, name="RL"))
    c.add(FaultyNmos("out", "in", "0", p, name="M1", fault=fault,
                     strength=strength))
    return c


def build_cases(seed: int = 0) -> List[Case]:
    """A balanced set: several instances of each class, at different biases."""
    rng = np.random.default_rng(seed)
    p = DeviceParams()
    cases: List[Case] = []

    for k, vin in enumerate((0.5, 0.7, 0.9, 1.1)):
        cases.append(Case(f"none_{k}", "none",
                          _inverter_with(p, vin, "none", 1.0)))

    # The ceiling is placed below the current the device would otherwise reach,
    # so the solve runs into it rather than past it.
    # The threshold is put near where the output settles, so the solve has to
    # cross it rather than sitting comfortably on one side.
    for k, (vin, vth) in enumerate(((0.85, 0.62), (0.95, 0.52),
                                    (1.05, 0.44), (1.15, 0.38))):
        cases.append(Case(f"kink_{k}", "c1_kink",
                          _inverter_with(p, vin, "c1_kink", vth)))

    for k, (vin, s) in enumerate(((0.9, 4.0), (1.0, 6.0),
                                  (1.1, 8.0), (1.2, 10.0))):
        cases.append(Case(f"neggds_{k}", "negative_gds",
                          _inverter_with(p, vin, "negative_gds", s)))

    for k, vin in enumerate((0.7, 0.9, 1.05, 1.2)):
        cases.append(Case(f"trunc_{k}", "truncated_inner_loop",
                          _inverter_with(p, vin, "truncated_inner_loop", 1.0)))

    # No defect: a healthy circuit started somewhere absurd.
    for k, vin in enumerate((0.6, 0.8, 1.0, 1.15)):
        c = _inverter_with(p, vin, "none", 1.0)
        n, m = c.size()
        x0 = np.zeros(n + m)
        nmap = c.node_map()
        x0[nmap["out"]] = float(rng.uniform(40.0, 80.0))
        cases.append(Case(f"badstart_{k}", "bad_initial_guess", c, x0=x0,
                          note="healthy circuit, start far outside the rails"))

    # Genuinely no solution: a current source demanding more than the device
    # can supply, with the output node otherwise unconstrained.
    for k, (vin, iload) in enumerate(((1.0, 9.0e-4), (1.1, 1.1e-3),
                                      (0.9, 8.0e-4), (1.2, 1.4e-3))):
        c = Circuit()
        c.add(VoltageSource("vdd", "0", 1.2, name="VDD"))
        c.add(VoltageSource("in", "0", vin, name="VIN"))
        c.add(CurrentSource("vdd", "out", iload, name="IL"))
        c.add(FaultyNmos("out", "in", "0", p, name="M1"))
        cases.append(Case(f"singular_{k}", "singular_operating_point", c,
                          note="load current exceeds what the device supplies"))

    return cases


def solve_case(case: Case, max_iter: int = 40) -> Dict:
    """Solve, then retry from nominal, and hand back only symptoms."""
    from .solver import solve_dc

    r = solve_dc(case.circuit, case.x0, max_iter=max_iter)
    trace = r.trace()

    # The retry is the diagnostic a person would reach for first: if it
    # converges from a sensible start, the circuit was fine and the start was
    # not.  It is a symptom, not the answer -- several faults also fail on
    # retry, and one of them fails for a reason that has nothing to do with
    # the model.
    retry = solve_dc(case.circuit, None, max_iter=max_iter)
    trace["retry_from_default_converged"] = bool(retry.converged)
    trace["retry_iterations"] = retry.iterations
    trace["started_far_from_default"] = bool(
        case.x0 is not None and float(np.max(np.abs(case.x0))) > 5.0)
    return trace
