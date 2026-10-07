"""The circuits themselves: transfer curve, mirror, and the ring oscillator.

The ring tests are slow because they are real transients.  Run the fast suite
with ``-m "not slow"``.
"""

from __future__ import annotations

import numpy as np
import pytest

from circuit.circuits import (VDD, mirror_error, propagation_delay,
                              transfer_curve)
from circuit.device import DeviceParams
from circuit.measure import CIRCUIT_METRICS, measure_device, measure_population


# --------------------------------------------------------------------------
# Transfer curve
# --------------------------------------------------------------------------

def test_transfer_curve_solves_everywhere(nominal):
    tc, traces = transfer_curve(nominal, n=15)
    assert tc.converged.all()
    assert len(traces) == 15


def test_transfer_curve_is_inverting(nominal):
    tc, _ = transfer_curve(nominal, n=15)
    assert np.all(np.diff(tc.vout) < 0.0)


def test_transfer_curve_stays_inside_the_rails(nominal):
    tc, _ = transfer_curve(nominal, n=15)
    assert np.all(tc.vout > 0.0) and np.all(tc.vout <= VDD + 1e-9)


def test_switching_threshold_is_inside_the_sweep(nominal):
    tc, _ = transfer_curve(nominal, n=25)
    vth = tc.switching_threshold()
    assert np.isfinite(vth)
    assert tc.vin[0] < vth < tc.vin[-1]


def test_gain_exceeds_the_three_stage_requirement(nominal):
    """2.18 against the 2.000 three stages need -- too thin, and measured.

    The README reports the 3-stage ring as not oscillating because of this
    margin, so the margin itself is worth pinning: if the gain ever fell below
    2.0, "too thin" would become "impossible" and the explanation changes.
    """
    tc, _ = transfer_curve(nominal, n=41)
    gain = tc.peak_gain()
    assert gain > 1.0 / np.cos(np.pi / 3), gain
    assert gain < 3.0, "gain this high would make the 3-stage story wrong"


def test_transfer_curve_and_quadrature_metrics_agree_on_the_dc_numbers(nominal):
    """Two independent paths to the same three DC metrics.

    ``transfer_curve`` walks the full Newton solve node by node;
    ``measure_device`` bisects the single-node equation.  They are different
    code, and the population results depend on the second one.
    """
    tc, _ = transfer_curve(nominal, n=25)
    m = measure_device(nominal)
    assert np.isclose(tc.switching_threshold(), m["vth_switch"], atol=5e-3)
    assert np.isclose(tc.v_low(), m["v_low"], atol=5e-3)
    assert np.isclose(tc.peak_gain(), m["gain"], rtol=0.1)


# --------------------------------------------------------------------------
# Current mirror
# --------------------------------------------------------------------------

def test_mirror_solves_and_reports_its_spread(nominal):
    r = mirror_error(nominal)
    assert r["all_converged"]
    assert len(r["currents"]) >= 3
    assert r["spread_percent"] > 0.0


def test_mirror_output_conductance_is_reported_with_its_sign(nominal):
    """A negative output resistance here would be the defect biting.

    It does not at this bias, and the flag is what says so rather than a
    silent absence.
    """
    r = mirror_error(nominal)
    assert isinstance(r["negative_output_resistance"], bool)
    assert np.isfinite(r["output_conductance"])


# --------------------------------------------------------------------------
# Population measurement
# --------------------------------------------------------------------------

def test_population_measurement_has_every_metric(nominal):
    devices = [DeviceParams(vth0=0.40 + 0.01 * i) for i in range(6)]
    pop = measure_population(devices)
    assert set(pop) == set(CIRCUIT_METRICS)
    for k, v in pop.items():
        assert v.shape == (len(devices),), k
        assert np.isfinite(v).all(), k


def test_slower_devices_have_longer_fall_delays():
    """Direction check on the metric stage 2's corners are scored against."""
    fast = measure_device(DeviceParams(vth0=0.36))
    slow = measure_device(DeviceParams(vth0=0.50))
    assert slow["t_fall"] > fast["t_fall"]
    assert slow["vth_switch"] > fast["vth_switch"]


# --------------------------------------------------------------------------
# Ring oscillator
# --------------------------------------------------------------------------

def test_ring_result_is_self_consistent():
    """``oscillated`` and ``stage_delay`` must never disagree.

    They are decided by different expressions -- ``period is not None`` and
    ``if period`` -- which differ when a period of exactly zero comes back.
    Nothing has produced that yet; this is what would catch it, instead of a
    report claiming a circuit oscillated with a stage delay of NaN.
    """
    for stages in (3,):
        d = propagation_delay(DeviceParams(), stages=stages)
        assert d["oscillated"] == bool(np.isfinite(d["stage_delay"]))
        assert d["oscillated"] == bool(np.isfinite(d["frequency"]))


@pytest.mark.slow
def test_three_stage_ring_does_not_oscillate(nominal):
    """Reported as not oscillating, and it has to stay a clean solve.

    If this ever failed to converge instead, the README's explanation -- gain
    margin too thin, not a solver problem -- would be wrong.
    """
    d = propagation_delay(nominal, stages=3)
    assert d["converged"]
    assert not d["oscillated"]


@pytest.mark.slow
def test_five_stage_ring_oscillates_with_a_plausible_stage_delay(nominal):
    """Stage delay is a property of the stage, so it must not track N."""
    d = propagation_delay(nominal, stages=5)
    assert d["converged"] and d["oscillated"]
    assert 10e-12 < d["stage_delay"] < 60e-12, d["stage_delay"]
    assert 1e9 < d["frequency"] < 1e10, d["frequency"]
