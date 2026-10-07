"""Axis 1 -- the device evaluation and its derivatives.

This is where stage 3 started, because project 12's series-resistance loop
returned an unconverged current and the derivatives did not belong to the
current they were paired with.  Every claim the rest of the stage makes about
convergence rests on these.
"""

from __future__ import annotations

import numpy as np
import pytest

from circuit.device import (I_FLOOR, P12_FIXED_POINT_CAP, DeviceParams,
                            current, evaluate, fixed_point_iterations,
                            reference_current)

GRID = [(vg, vd) for vg in (0.5, 0.8, 1.0, 1.2) for vd in (0.1, 0.4, 0.8, 1.2)]


def _fd(p: DeviceParams, vg: float, vd: float, which: str,
        h: float = 1e-6) -> float:
    """Central difference of the terminal current."""
    if which == "gm":
        a = current(p, np.array([vg + h]), np.array([vd]))[0]
        b = current(p, np.array([vg - h]), np.array([vd]))[0]
    else:
        a = current(p, np.array([vg]), np.array([vd + h]))[0]
        b = current(p, np.array([vg]), np.array([vd - h]))[0]
    return float((a - b) / (2 * h))


# --------------------------------------------------------------------------
# The derivatives belong to the current
# --------------------------------------------------------------------------

@pytest.mark.parametrize("vg,vd", GRID)
def test_gm_matches_finite_difference(nominal, vg, vd):
    ev = evaluate(nominal, np.array([vg]), np.array([vd]))
    assert ev.converged
    num = _fd(nominal, vg, vd, "gm")
    assert np.isclose(float(ev.gm[0]), num, rtol=2e-5, atol=1e-16)


@pytest.mark.parametrize("vg,vd", GRID)
def test_gds_matches_finite_difference(nominal, vg, vd):
    ev = evaluate(nominal, np.array([vg]), np.array([vd]))
    assert ev.converged
    num = _fd(nominal, vg, vd, "gds")
    assert np.isclose(float(ev.gds[0]), num, rtol=2e-5, atol=1e-16)


@pytest.mark.parametrize("rs", [0.0, 50.0, 500.0, 2000.0])
def test_derivatives_hold_across_series_resistance(rs):
    """The implicit function theorem is the whole point of the rewrite.

    At ``rs = 0`` the terminal derivative is the intrinsic one; as ``rs``
    grows, ``1 + f_d*rs`` divides it down.  Both ends have to match a finite
    difference of the current the solve actually returned.
    """
    p = DeviceParams(rs=rs)
    vg, vd = 1.1, 0.9
    ev = evaluate(p, np.array([vg]), np.array([vd]))
    assert ev.converged
    assert np.isclose(float(ev.gm[0]), _fd(p, vg, vd, "gm"),
                      rtol=1e-4, atol=1e-15)
    assert np.isclose(float(ev.gds[0]), _fd(p, vg, vd, "gds"),
                      rtol=1e-4, atol=1e-15)


def test_series_resistance_degrades_gm_only_where_gds_is_positive():
    """The textbook direction holds only on the healthy side of the kink.

    ``gm = f_g / (1 + f_d*rs)``.  Where the intrinsic output conductance
    ``f_d`` is positive the denominator exceeds one and series resistance
    degrades transconductance, as everyone expects.  Where ``f_d`` is
    negative -- which this model reaches -- the denominator drops below one
    and series resistance *amplifies* gm instead.  The first version of this
    test asserted the textbook direction everywhere and failed, which is how
    the asymmetry got measured.
    """
    def gm_at(vg, vd, rs):
        return float(evaluate(DeviceParams(rs=rs), np.array([vg]),
                              np.array([vd])).gm[0])

    # Healthy region: positive intrinsic gds, gm degrades.
    assert 0.0 < gm_at(0.8, 0.5, 500.0) < gm_at(0.8, 0.5, 0.0)
    # Turned-over region: negative intrinsic gds, gm is amplified.
    assert gm_at(1.2, 1.2, 500.0) > gm_at(1.2, 1.2, 0.0) > 0.0


def test_series_resistance_can_flip_the_sign_of_terminal_gds():
    """Series resistance partially masks the model's own defect.

    At this bias the intrinsic device has ``gds < 0`` and the terminal device,
    seen through ``rs``, has ``gds > 0``.  So the fraction of the grid that
    looks turned-over depends on which ``rs`` you ask about, and the figure
    the report quotes is the one at nominal.
    """
    vg, vd = np.array([1.1]), np.array([0.9])
    assert float(evaluate(DeviceParams(rs=0.0), vg, vd).gds[0]) < 0.0
    assert float(evaluate(DeviceParams(rs=500.0), vg, vd).gds[0]) > 0.0


def test_series_resistance_shrinks_the_negative_region_monotonically():
    """Measured: 15.3% at rs=0, 12.7% at nominal 180, 8.3% at 500.

    Pinned as an ordering rather than three numbers.  This is the second
    reason no solve broke on the defect, alongside the load-line argument:
    the circuit runs at nominal rs, where the region is a fifth smaller than
    the model's own.
    """
    vg = np.linspace(0.4, 1.2, 31)
    vd = np.linspace(0.05, 1.2, 31)
    VG, VD = np.meshgrid(vg, vd, indexing="ij")

    def fraction(rs):
        g = evaluate(DeviceParams(rs=rs), VG.ravel(), VD.ravel()).gds
        return float((g < 0.0).mean())

    f0, f_nom, f_hi = fraction(0.0), fraction(DeviceParams().rs), fraction(500.)
    assert f0 > f_nom > f_hi > 0.0


# --------------------------------------------------------------------------
# Job 0: the fixed-point loop, and the margin the report quotes
# --------------------------------------------------------------------------

def test_project12_cap_is_still_forty():
    """The margin figures are only meaningful while the cap is what we say.

    Project 12 writes the cap as a bare ``range(40)`` rather than a named
    constant, so nothing in its API would tell us if it moved.  Reading the
    source is ugly and it is the only honest way to pin it.
    """
    import cmext.model as m
    from pathlib import Path
    src = Path(m.__file__).read_text(encoding="utf-8")
    assert f"range({P12_FIXED_POINT_CAP})" in src, (
        "project 12's fixed-point cap changed; every iteration margin this "
        "stage reports about it is now stale")


def test_fixed_point_needs_most_of_its_budget_at_nominal(nominal):
    """38 of 40 at full drive -- the finding, pinned as a range not a point.

    Pinned loosely on purpose: the exact count depends on the convergence
    threshold and would make this a brittle test of arithmetic.  What matters
    is that it is close enough to the cap to be alarming and still under it.
    """
    n = fixed_point_iterations(nominal, np.array([1.2]), np.array([1.2]))
    assert 30 <= n < P12_FIXED_POINT_CAP


def test_fixed_point_blows_the_budget_at_high_series_resistance():
    hard = DeviceParams(rs=2000.0, mu0=0.09)
    n = fixed_point_iterations(hard, np.array([1.2]), np.array([1.2]),
                               max_iter=500)
    assert n is None or n > P12_FIXED_POINT_CAP


def test_newton_converges_where_fixed_point_does_not():
    hard = DeviceParams(rs=2000.0, mu0=0.09)
    ev = evaluate(hard, np.array([1.2]), np.array([1.2]))
    assert ev.converged
    assert ev.iterations < 20


def test_project12_error_at_high_series_resistance_is_large():
    """The 72% claim, pinned as "badly wrong" rather than as a number."""
    hard = DeviceParams(rs=2000.0, mu0=0.09)
    vg, vd = np.array([1.2]), np.array([1.2])
    new = float(evaluate(hard, vg, vd).ids[0])
    old = float(reference_current(hard, vg, vd)[0])
    assert abs(new - old) / new > 0.2


def test_physics_unchanged_where_project12_converged(nominal):
    """The rewrite must not have moved the model, only resolved it.

    Away from high drive project 12's loop does converge, and there the two
    have to agree to solver tolerance -- otherwise stage 2's results, which
    were produced with the old evaluation, would need redoing.
    """
    vg = np.array([0.6, 0.7, 0.8])
    vd = np.array([0.2, 0.3, 0.4])
    new = evaluate(nominal, vg, vd).ids
    old = reference_current(nominal, vg, vd)
    assert np.allclose(new, old, rtol=1e-9)


# --------------------------------------------------------------------------
# Negative output conductance: the defect this stage found
# --------------------------------------------------------------------------

def test_negative_gds_exists_in_the_model(nominal):
    vg = np.full(40, 1.2)
    vd = np.linspace(0.05, 1.2, 40)
    gds = evaluate(nominal, vg, vd).gds
    assert np.any(gds < 0.0), "the kink this stage reports has disappeared"


def test_negative_gds_is_intrinsic_not_a_series_resistance_artefact():
    """With ``rs = 0`` the denominator is 1, so anything negative is the model.

    This is the difference between "our new derivative is wrong" and "the
    compact model turns over", and it is the reason the finding is reported
    as a model defect.
    """
    p = DeviceParams(rs=0.0)
    vg = np.full(40, 1.2)
    vd = np.linspace(0.05, 1.2, 40)
    assert np.any(evaluate(p, vg, vd).gds < 0.0)


def test_current_turns_over_at_high_drive(nominal):
    """The same defect read off the current, independent of any derivative."""
    vd = np.linspace(0.6, 1.2, 25)
    ids = evaluate(nominal, np.full_like(vd, 1.2), vd).ids
    assert ids[-1] < ids.max()


# --------------------------------------------------------------------------
# Shape, floor, and the plumbing
# --------------------------------------------------------------------------

def test_current_stays_positive_deep_in_subthreshold(nominal):
    ids = evaluate(nominal, np.array([0.0, 0.05]), np.array([0.5, 0.5])).ids
    assert np.all(ids >= I_FLOOR * 0.5)
    assert np.all(np.isfinite(ids))


def test_scalar_and_array_inputs_agree(nominal):
    a = evaluate(nominal, 1.0, 0.6)
    b = evaluate(nominal, np.array([1.0]), np.array([0.6]))
    assert np.isclose(float(a.ids[0]), float(b.ids[0]), rtol=1e-14)


def test_broadcasting_matches_elementwise(nominal):
    vd = np.array([0.2, 0.6, 1.0])
    broadcast = evaluate(nominal, 1.0, vd).ids
    one_at_a_time = np.array([float(evaluate(nominal, 1.0, v).ids[0])
                              for v in vd])
    assert np.allclose(broadcast, one_at_a_time, rtol=1e-12)
