"""Axis 4 -- element stamps, residual signs, and the Jacobian.

Two of this stage's bugs lived here and neither announced itself: the current
source pushed current the wrong way, so the node had two sinks and no source
and solves failed although solutions existed; and the capacitor had to stay
inert outside a transient.  A sign error in a stamp does not raise -- it
produces a wrong answer or a mysterious non-convergence, so the stamps are
checked against hand-solvable circuits and the Jacobian against a finite
difference of the residual it claims to be the derivative of.
"""

from __future__ import annotations

import numpy as np
import pytest

from circuit.device import DeviceParams
from circuit.netlist import (Capacitor, Circuit, CurrentSource, Nmos,
                             Resistor, VoltageSource)
from circuit.solver import solve_dc


def _jacobian_fd(c: Circuit, x: np.ndarray, h: float = 1e-7) -> np.ndarray:
    n = x.size
    J = np.zeros((n, n))
    for k in range(n):
        xp, xm = x.copy(), x.copy()
        xp[k] += h
        xm[k] -= h
        J[:, k] = (c.residual(xp)[0] - c.residual(xm)[0]) / (2 * h)
    return J


# --------------------------------------------------------------------------
# Hand-solvable circuits: the stamps mean what they say
# --------------------------------------------------------------------------

def test_resistive_divider():
    c = Circuit()
    c.add(VoltageSource("top", "0", 1.0, name="V1"))
    c.add(Resistor("top", "mid", 1.0e3, name="R1"))
    c.add(Resistor("mid", "0", 3.0e3, name="R2"))
    r = solve_dc(c)
    assert r.converged
    assert np.isclose(c.voltage(r.x, "mid"), 0.75, rtol=1e-9)


def test_voltage_source_sets_its_node():
    c = Circuit()
    c.add(VoltageSource("a", "0", 0.42, name="V1"))
    c.add(Resistor("a", "0", 1.0e3, name="R1"))
    r = solve_dc(c)
    assert r.converged
    assert np.isclose(c.voltage(r.x, "a"), 0.42, rtol=1e-12)


def test_current_source_pushes_current_from_a_to_b():
    """The sign bug, pinned.

    ``CurrentSource(a, b, i)`` forces ``i`` out of ``a`` and into ``b``.  With
    ``b`` tied to ground through a resistor the node must sit at ``+i*R``.
    When this was backwards the device and the load both sank current from the
    output node, and solves that had perfectly good solutions diverged.
    """
    i, rl = 1.0e-3, 2.0e3
    c = Circuit()
    c.add(CurrentSource("0", "out", i, name="I1"))
    c.add(Resistor("out", "0", rl, name="RL"))
    r = solve_dc(c)
    assert r.converged
    assert np.isclose(c.voltage(r.x, "out"), i * rl, rtol=1e-9)


def test_current_source_reversed_gives_the_opposite_sign():
    i, rl = 1.0e-3, 2.0e3
    c = Circuit()
    c.add(CurrentSource("out", "0", i, name="I1"))
    c.add(Resistor("out", "0", rl, name="RL"))
    r = solve_dc(c)
    assert r.converged
    assert np.isclose(c.voltage(r.x, "out"), -i * rl, rtol=1e-9)


def test_nmos_sinks_current_from_its_drain():
    """A conducting NMOS must pull its drain *down*, never up."""
    c = Circuit()
    c.add(VoltageSource("vdd", "0", 1.2, name="VDD"))
    c.add(VoltageSource("in", "0", 1.0, name="VIN"))
    c.add(Resistor("vdd", "out", 2.0e3, name="RL"))
    c.add(Nmos("out", "in", "0", DeviceParams(), name="M1"))
    r = solve_dc(c)
    assert r.converged
    vout = c.voltage(r.x, "out")
    assert 0.0 < vout < 1.2


def test_inverter_output_falls_as_input_rises():
    p = DeviceParams()
    levels = []
    for vin in (0.4, 0.7, 1.0, 1.2):
        c = Circuit()
        c.add(VoltageSource("vdd", "0", 1.2, name="VDD"))
        c.add(VoltageSource("in", "0", vin, name="VIN"))
        c.add(Resistor("vdd", "out", 2.0e3, name="RL"))
        c.add(Nmos("out", "in", "0", p, name="M1"))
        r = solve_dc(c)
        assert r.converged
        levels.append(c.voltage(r.x, "out"))
    assert all(a > b for a, b in zip(levels, levels[1:])), levels


# --------------------------------------------------------------------------
# The Jacobian is the derivative of the residual
# --------------------------------------------------------------------------

def test_jacobian_matches_finite_difference_linear():
    c = Circuit()
    c.add(VoltageSource("top", "0", 1.0, name="V1"))
    c.add(Resistor("top", "mid", 1.0e3, name="R1"))
    c.add(Resistor("mid", "0", 3.0e3, name="R2"))
    x = np.array([0.9, 0.5, -1e-4])
    _, J = c.residual(x)
    assert np.allclose(J, _jacobian_fd(c, x), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("vin", [0.5, 0.8, 1.1])
def test_jacobian_matches_finite_difference_with_device(vin):
    """The strongest single check in this file.

    The device stamp writes ``gm`` and ``gds`` into three Jacobian entries
    with three different signs.  Any one of them wrong still converges
    sometimes, which is how such a bug survives.  Comparing the whole matrix
    against a difference of the residual catches all of them at once.
    """
    c = Circuit()
    c.add(VoltageSource("vdd", "0", 1.2, name="VDD"))
    c.add(VoltageSource("in", "0", vin, name="VIN"))
    c.add(Resistor("vdd", "out", 2.0e3, name="RL"))
    c.add(Nmos("out", "in", "0", DeviceParams(), name="M1"))
    r = solve_dc(c)
    assert r.converged
    _, J = c.residual(r.x)
    fd = _jacobian_fd(c, r.x, h=1e-7)
    scale = max(1.0, float(np.max(np.abs(fd))))
    assert np.allclose(J, fd, rtol=1e-4, atol=1e-6 * scale)


def test_residual_vanishes_at_the_solution():
    c = Circuit()
    c.add(VoltageSource("vdd", "0", 1.2, name="VDD"))
    c.add(VoltageSource("in", "0", 0.9, name="VIN"))
    c.add(Resistor("vdd", "out", 2.0e3, name="RL"))
    c.add(Nmos("out", "in", "0", DeviceParams(), name="M1"))
    r = solve_dc(c)
    assert r.converged
    F, _ = c.residual(r.x)
    assert float(np.max(np.abs(F))) < 1e-10


# --------------------------------------------------------------------------
# Capacitor: inert in DC
# --------------------------------------------------------------------------

def test_capacitor_is_inert_without_a_timestep():
    """DC must not see the capacitor at all, or every DC level shifts."""
    def level(with_cap: bool) -> float:
        c = Circuit()
        c.add(VoltageSource("vdd", "0", 1.2, name="VDD"))
        c.add(VoltageSource("in", "0", 0.9, name="VIN"))
        c.add(Resistor("vdd", "out", 2.0e3, name="RL"))
        c.add(Nmos("out", "in", "0", DeviceParams(), name="M1"))
        if with_cap:
            c.add(Capacitor("out", "0", 20e-15, name="CL"))
        r = solve_dc(c)
        assert r.converged
        return c.voltage(r.x, "out")

    assert np.isclose(level(True), level(False), rtol=1e-12)


def test_capacitor_stamps_once_a_timestep_is_set():
    c = Circuit()
    c.add(VoltageSource("a", "0", 1.0, name="V1"))
    cap = Capacitor("a", "0", 20e-15, name="C1")
    c.add(cap)
    x = np.array([0.5, 0.0])
    F_before, _ = c.residual(x)
    cap.dt = 1e-13
    cap.v_old = 0.0
    F_after, _ = c.residual(x)
    assert not np.allclose(F_before, F_after)


def test_ground_is_not_a_variable():
    c = Circuit()
    c.add(Resistor("a", "0", 1.0e3, name="R1"))
    assert c.node_map()["0"] == -1
    n, m = c.size()
    assert n == 1 and m == 0
