"""Tests for the claims this project makes.

    python tests/test_cmext.py      # no dependencies beyond the package
    pytest tests/test_cmext.py      # if pytest is installed

The emphasis is on the physics being right and the gates actually gating.
Numerical outputs are allowed to move; the invariants are not.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from cmext.extraction import (extract_simultaneous, extract_staged,
                              recovery_error, seed_params)
from cmext.identifiability import analyse, log_jacobian
from cmext.model import (BOUNDS, LN10, PHI_T, DeviceParams, drain_current,
                         measure, sweep_grid)

TRUTH = DeviceParams()
VG, VD = sweep_grid()


# --------------------------------------------------------------------------- #
# Model physics
# --------------------------------------------------------------------------- #

def test_subthreshold_swing_matches_parameter():
    """The extracted SS of the generated data must equal the SS we set."""
    ids = drain_current(TRUTH, VG, VD)
    m = VD == VD.min()
    g, i = VG[m], ids[m]
    sel = (i > 1e-11) & (i < 1e-8)
    ss = 1.0 / np.polyfit(g[sel], np.log10(i[sel]), 1)[0]
    assert abs(ss - TRUTH.ss) / TRUTH.ss < 0.02, f"SS {ss*1000:.1f} vs {TRUTH.ss*1000:.1f} mV/dec"


def test_dibl_shift_matches_eta():
    """Vth must fall by eta * dVds between the low and high Vds sweeps."""
    ids = drain_current(TRUTH, VG, VD)

    def vth_cc(vd_val, level=1e-7):
        m = VD == vd_val
        o = np.argsort(ids[m])
        return float(np.interp(np.log10(level), np.log10(ids[m][o]), VG[m][o]))

    shift = vth_cc(VD.min()) - vth_cc(VD.max())
    expect = TRUTH.eta * (VD.max() - VD.min())
    assert abs(shift - expect) / expect < 0.05, f"DIBL {shift:.4f} vs {expect:.4f}"


def test_current_monotonic_in_vgs():
    ids = drain_current(TRUTH, VG, VD)
    for v in np.unique(VD):
        s = ids[VD == v]
        assert np.all(np.diff(s) > 0), f"Id not monotonic in Vgs at Vds={v}"


def test_subthreshold_swing_respects_thermal_floor():
    """No parameter set may produce a swing below the 300 K limit."""
    assert BOUNDS["ss"][0] >= LN10 * PHI_T * 0.999


def test_series_resistance_reduces_current():
    hi_rs = TRUTH.copy_with(rs=TRUTH.rs * 4)
    a = drain_current(TRUTH, np.array([1.2]), np.array([0.05]))
    b = drain_current(hi_rs, np.array([1.2]), np.array([0.05]))
    assert b[0] < a[0], "more series resistance must give less current"


# --------------------------------------------------------------------------- #
# Extraction and gates
# --------------------------------------------------------------------------- #

def test_noise_free_recovery_is_accurate():
    """With clean data the simultaneous fit must recover the truth closely."""
    ids = drain_current(TRUTH, VG, VD)
    r = extract_simultaneous(VG, VD, ids)
    err = recovery_error(TRUTH, r.params)
    worst = max(err.values())
    assert worst < 0.10, f"worst recovery error {worst*100:.1f}%: {err}"


def test_gate_rejects_unphysical_swing():
    """A sub-thermal subthreshold swing must fail the gate, not be accepted."""
    from cmext.extraction import _physics_gates
    gates = _physics_gates("subthreshold", {"ss": 0.030, "vth0": 0.4}, rms=0.001)
    named = {g.name: g for g in gates}
    assert not named["ss_thermal_floor"].passed


def test_gate_flags_parameter_pinned_at_bound():
    """A value sitting on its bound means the data did not determine it."""
    from cmext.extraction import _bound_gates
    g = {x.name: x for x in _bound_gates({"theta": BOUNDS["theta"][0]})}
    assert not g["bounds[theta]"].passed


def test_staged_escalates_rather_than_returning_untrustworthy_theta():
    """theta is not identifiable from this measurement set.

    The staged extraction must say so by escalating, instead of handing back a
    number that the noise sweep shows is unreliable.
    """
    ids = measure(TRUTH, VG, VD, noise_rel=0.01, seed=3)
    r = extract_staged(VG, VD, ids)
    assert r.escalated, "staged run should escalate when a gate fails"


def test_staged_recovers_the_well_conditioned_parameters():
    """Escalating must not mean giving up: the strong stages still deliver."""
    ids = measure(TRUTH, VG, VD, noise_rel=0.01, seed=4)
    r = extract_staged(VG, VD, ids)
    err = recovery_error(TRUTH, r.params)
    assert err["ss"] < 0.05, f"ss error {err['ss']*100:.1f}%"
    assert err["eta"] < 0.10, f"eta error {err['eta']*100:.1f}%"


# --------------------------------------------------------------------------- #
# Identifiability
# --------------------------------------------------------------------------- #

def test_jacobian_is_finite_and_nonzero_for_every_parameter():
    J, names = log_jacobian(TRUTH, VG, VD)
    assert np.all(np.isfinite(J))
    for i, n in enumerate(names):
        assert np.max(np.abs(J[:, i])) > 0, f"{n} has no effect on the data"


def test_predicted_uncertainty_matches_monte_carlo():
    """The headline claim: the Jacobian predicts the real spread.

    If this fails, the identifiability report is decoration rather than a
    usable estimate, and nothing else in the project should be trusted.
    """
    rep = analyse(TRUTH, VG, VD, noise_rel=0.01)
    ests = {n: [] for n in DeviceParams.names()}
    for s in range(8):
        ids = measure(TRUTH, VG, VD, noise_rel=0.01, seed=200 + s)
        r = extract_simultaneous(VG, VD, ids)
        for n, v in r.params.to_dict().items():
            ests[n].append(v)

    for n in ("vth0", "ss", "mu0", "rs", "eta"):
        a = np.array(ests[n])
        measured = a.std() / abs(a.mean())
        predicted = rep.rel_std_error[n]
        ratio = measured / max(predicted, 1e-12)
        assert 0.3 < ratio < 3.0, (
            f"{n}: predicted {predicted*100:.2f}%, measured {measured*100:.2f}%")


def test_least_sensitive_parameter_is_the_one_that_breaks():
    """Sensitivity ranking must agree with which parameter is least certain."""
    rep = analyse(TRUTH, VG, VD, noise_rel=0.01)
    least_sensitive = min(rep.sensitivity, key=rep.sensitivity.get)
    most_uncertain = max(rep.rel_std_error, key=rep.rel_std_error.get)
    assert least_sensitive in ("theta", "rs", "vsat")
    assert most_uncertain in ("theta", "rs", "vsat")


def test_strong_correlation_is_detected():
    """vth0 and mu0 trade off in the linear region; the report must say so."""
    rep = analyse(TRUTH, VG, VD, noise_rel=0.01, corr_threshold=0.90)
    pairs = {frozenset((a, b)) for a, b, _ in rep.strong_pairs}
    assert frozenset(("vth0", "mu0")) in pairs, f"pairs found: {rep.strong_pairs}"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
