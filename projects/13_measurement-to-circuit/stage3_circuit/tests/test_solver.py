"""Axis 2 -- the solver, the transient, and the quadrature substitution.

The transient bug in this stage was silent in the worst way: ``solve_transient``
re-solved DC after the input had already been stepped, so the circuit started
at its *final* operating point, nothing moved, and the delay came back NaN.
Nothing raised.  The tests here pin the initial condition explicitly, and then
check the quadrature delay -- which the population measurement actually uses --
against the Newton transient it stands in for.
"""

from __future__ import annotations

import numpy as np
import pytest

from circuit.circuits import C_LOAD, R_LOAD, VDD, ring_oscillator
from circuit.device import DeviceParams
from circuit.measure import (dc_level, dc_levels, fall_delay,
                             transient_fall_delay, validate_quadrature)
from circuit.netlist import (Capacitor, Circuit, Nmos, Resistor,
                             VoltageSource)
from circuit.solver import solve_dc, solve_transient


def _stepped_inverter(p: DeviceParams):
    c = Circuit()
    c.add(VoltageSource("vdd", "0", VDD, name="VDD"))
    c.add(VoltageSource("in", "0", 0.0, name="VIN"))
    c.add(Resistor("vdd", "out", R_LOAD, name="RL"))
    c.add(Nmos("out", "in", "0", p, name="M1"))
    c.add(Capacitor("out", "0", C_LOAD, name="CL"))
    dc = solve_dc(c)
    assert dc.converged
    vin = [e for e in c.elements if getattr(e, "name", "") == "VIN"][0]
    vin.v = VDD
    return c, dc


# --------------------------------------------------------------------------
# The supplied state is the initial condition, not a starting guess
# --------------------------------------------------------------------------

def test_transient_starts_from_the_supplied_state(nominal):
    """The bug that returned no delay at all.

    The input is stepped *after* the DC solve, so the state handed in belongs
    to the pre-step circuit.  If the transient re-solves DC it lands on the
    post-step operating point and there is no transition left to simulate.
    """
    c, dc = _stepped_inverter(nominal)
    v_high = c.voltage(dc.x, "out")
    r = solve_transient(c, t_stop=3e-10, dt=1e-13, x0=dc.x)
    assert r.converged
    assert np.isclose(r.v["out"][0], v_high, rtol=1e-9), (
        "transient did not start where it was told to")
    assert r.v["out"][-1] < 0.5 * v_high, "output never fell"


def test_transient_without_x0_settles_and_stays(nominal):
    """No initial condition means start at DC, where nothing should move."""
    c = Circuit()
    c.add(VoltageSource("vdd", "0", VDD, name="VDD"))
    c.add(VoltageSource("in", "0", 1.0, name="VIN"))
    c.add(Resistor("vdd", "out", R_LOAD, name="RL"))
    c.add(Nmos("out", "in", "0", nominal, name="M1"))
    c.add(Capacitor("out", "0", C_LOAD, name="CL"))
    r = solve_transient(c, t_stop=1e-10, dt=1e-13)
    assert r.converged
    v = r.v["out"]
    assert np.allclose(v, v[0], atol=1e-9), "a settled circuit drifted"


def test_transient_output_is_monotonic_while_falling(nominal):
    c, dc = _stepped_inverter(nominal)
    r = solve_transient(c, t_stop=2e-10, dt=1e-13, x0=dc.x)
    assert r.converged
    v = r.v["out"]
    assert np.all(np.diff(v) <= 1e-12), "backward Euler produced ringing"


# --------------------------------------------------------------------------
# Quadrature against the solver: the substitution the population rests on
# --------------------------------------------------------------------------

def test_quadrature_delay_matches_the_transient(nominal):
    """Measured at 0.15% for the nominal device; pinned at 1%.

    The population fall delay is a quadrature of the single-node equation
    rather than a Newton transient, because 400 transients is not affordable.
    That is a different integration of the same circuit's own equation, not an
    approximation of it -- so it has to agree, and the agreement is measured
    rather than asserted.
    """
    quad = fall_delay(nominal)
    solver = transient_fall_delay(nominal)
    assert np.isfinite(quad) and np.isfinite(solver)
    assert abs(quad - solver) / solver < 0.01


def test_quadrature_validation_over_several_devices():
    devices = [DeviceParams(vth0=v, mu0=m)
               for v, m in ((0.40, 0.10), (0.44, 0.09), (0.36, 0.11))]
    chk = validate_quadrature(devices)
    assert chk.n == len(devices)
    assert chk.max_rel_error < 0.02


def test_fall_delay_is_nan_when_the_device_cannot_reach_mid():
    """A weak device genuinely has no fall delay; it must not get a number.

    Returning something plausible here would have silently corrupted the
    population statistics with a made-up value.
    """
    weak = DeviceParams(vth0=1.6, mu0=0.002)
    assert np.isnan(fall_delay(weak))


# --------------------------------------------------------------------------
# The vectorised bisection is the same arithmetic
# --------------------------------------------------------------------------

def test_vectorised_dc_levels_match_one_at_a_time(nominal):
    """17x faster; the test is that it is also the same answer."""
    vins = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    batch = dc_levels(nominal, vins)
    single = np.array([dc_level(nominal, float(v)) for v in vins])
    ok = np.isfinite(batch) & np.isfinite(single)
    assert ok.sum() >= 4
    assert np.allclose(batch[ok], single[ok], atol=1e-6)


def test_dc_levels_are_monotonic_in_input(nominal):
    v = dc_levels(nominal, np.linspace(0.2, 1.2, 15))
    v = v[np.isfinite(v)]
    assert np.all(np.diff(v) < 0.0)


# --------------------------------------------------------------------------
# Solver behaviour and the trace contract
# --------------------------------------------------------------------------

def test_solver_reports_failure_rather_than_a_wrong_answer():
    """An impossible load: more current demanded than the device supplies."""
    from circuit.netlist import CurrentSource
    c = Circuit()
    c.add(VoltageSource("vdd", "0", VDD, name="VDD"))
    c.add(VoltageSource("in", "0", 1.0, name="VIN"))
    c.add(CurrentSource("vdd", "out", 9.0e-4, name="IL"))
    c.add(Nmos("out", "in", "0", DeviceParams(), name="M1"))
    r = solve_dc(c, max_iter=40)
    assert not r.converged
    assert r.failure, "a failed solve must name its failure mode"


def test_no_trace_key_is_named_after_a_fault_class():
    """The evidence must be symptoms, not labels -- including in its keys.

    This failed when first written: the trace carried a counter called
    ``devices_with_negative_gds``, which is verbatim one of the six class
    names.  Nothing read it and the arms' evidence never included it, so no
    measured result depended on it, but the claim "no field names the fault"
    was false and the key was there for anything that later reached for it.
    It is now named for the symptom.
    """
    from circuit.faults import FAULT_CLASSES
    c = Circuit()
    c.add(VoltageSource("vdd", "0", VDD, name="VDD"))
    c.add(VoltageSource("in", "0", 0.9, name="VIN"))
    c.add(Resistor("vdd", "out", R_LOAD, name="RL"))
    c.add(Nmos("out", "in", "0", DeviceParams(), name="M1"))
    trace = solve_dc(c).trace()

    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from keys(v)

    bad = [(k, cls) for k in keys(trace)
           for cls in FAULT_CLASSES if cls != "none" and cls in str(k)]
    assert not bad, f"trace keys named after fault classes: {bad}"


def test_arm_evidence_never_contains_a_class_name():
    """The stronger invariant: what the LLM is shown, keys and values."""
    from circuit.diagnose import evidence_for
    from circuit.faults import FAULT_CLASSES, build_cases, solve_case
    case = next(c for c in build_cases() if c.name == "neggds_0")
    ev = evidence_for(solve_case(case))
    flat = str(ev)
    for cls in FAULT_CLASSES:
        if cls == "none":
            continue
        assert cls not in flat, f"evidence leaks the label {cls!r}"


def test_converged_trace_has_a_shrinking_residual():
    c = Circuit()
    c.add(VoltageSource("vdd", "0", VDD, name="VDD"))
    c.add(VoltageSource("in", "0", 0.9, name="VIN"))
    c.add(Resistor("vdd", "out", R_LOAD, name="RL"))
    c.add(Nmos("out", "in", "0", DeviceParams(), name="M1"))
    t = solve_dc(c).trace()
    assert t["converged"]
    assert t["final_residual"] < 1e-10
    assert not t["diverging"]


def test_a_healthy_circuit_fails_from_a_far_start_but_not_from_default(nominal):
    """What makes ``bad_initial_guess`` a diagnosable class at all.

    Written first as "step limiting walks it back in", which is false -- 60 V
    on the output node does not recover even in 200 iterations.  That is the
    premise of the fault class: the circuit is fine, the start is not, and
    the retry from default is what separates them.  If step limiting ever got
    strong enough to recover, four cases would silently stop being diagnosable.
    """
    c = Circuit()
    c.add(VoltageSource("vdd", "0", VDD, name="VDD"))
    c.add(VoltageSource("in", "0", 0.9, name="VIN"))
    c.add(Resistor("vdd", "out", R_LOAD, name="RL"))
    c.add(Nmos("out", "in", "0", nominal, name="M1"))
    n, m = c.size()
    x0 = np.zeros(n + m)
    x0[c.node_map()["out"]] = 60.0

    assert not solve_dc(c, x0, max_iter=40).converged
    assert solve_dc(c, None, max_iter=40).converged


def test_step_limiting_caps_the_first_move(nominal):
    """No node may move more than ``vstep_max`` in one iteration."""
    from circuit.solver import VSTEP_MAX
    c = Circuit()
    c.add(VoltageSource("vdd", "0", VDD, name="VDD"))
    c.add(VoltageSource("in", "0", 0.9, name="VIN"))
    c.add(Resistor("vdd", "out", R_LOAD, name="RL"))
    c.add(Nmos("out", "in", "0", nominal, name="M1"))
    n, m = c.size()
    x0 = np.zeros(n + m)
    x0[c.node_map()["out"]] = 8.0
    r = solve_dc(c, x0, max_iter=60)
    first = r.history[0]
    assert first.damping * first.step_norm <= VSTEP_MAX * np.sqrt(n + m) + 1e-9


# --------------------------------------------------------------------------
# Ring oscillator: the gain condition decides which stage counts can run
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stages,needed", [(3, 2.0), (5, 1.236), (7, 1.110)])
def test_ring_gain_threshold(stages, needed):
    """``|A| > 1 / cos(pi/N)``.  Three stages need gain 2.000.

    The inverter's gain is about 2.18, which clears three stages by too little
    to oscillate in practice -- the reason the 3-stage ring is reported as not
    oscillating rather than as a solver failure.
    """
    assert np.isclose(1.0 / np.cos(np.pi / stages), needed, rtol=1e-3)


def test_ring_oscillator_builds_with_the_right_node_count(nominal):
    c = ring_oscillator(nominal, stages=5)
    inner = [n for n in c.nodes if n not in ("0", "vdd")]
    assert len(inner) >= 5
