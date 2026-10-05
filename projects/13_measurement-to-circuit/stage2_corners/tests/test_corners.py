"""Tests for stage 2.

The ones that matter check the claims the report makes, not that functions
run: that the sampler matches the covariance its own sensitivities imply, that
process and estimation correlation are kept apart, that the deconvolution
recovers a known spread for parameters that are measurable and is honest about
the ones that are not, and that each corner method fails in the direction its
docstring says it does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corners.deconvolve import (deconvolve, extraction_covariance,
                                project_psd)
from corners.device import DeviceParams, NoiseModel, PARAM_NAMES
from corners.evaluate import evaluate, nominal_coverage, reference_range
from corners.generate import (independent_box, one_at_a_time, pca_corners,
                              statistical_mc, worst_case_distance)
from corners.metrics import all_specs, evaluate_all, spec
from corners.physical import (LOCAL_SIGMA, SENSITIVITY, SS_FLOOR, anisotropy,
                              default_variables, process_covariance,
                              sample_population)

NOMINAL = DeviceParams()
NOISE = NoiseModel()


# ------------------------------------------------------------------ physical

def test_sampler_matches_its_own_analytic_covariance():
    """Sigma = S diag(sigma_x^2) S^T, computed two ways.

    Compared with a tolerance scaled to the matrix as a whole: a relative
    tolerance is meaningless on the near-zero off-diagonals.
    """
    pop = sample_population(20000, seed=3)
    analytic = process_covariance()
    scale = float(np.max(np.abs(analytic)))
    assert np.allclose(pop.covariance(), analytic, rtol=0.05,
                       atol=0.03 * scale)


def test_covariance_is_anisotropic_not_isotropic():
    """The parameters do not fill seven independent directions.

    Shared causes concentrate the variance into a few directions. Local
    per-parameter variation keeps the matrix full rank -- a real population is
    never exactly singular -- but most of the variance still lives in a
    handful of axes, which is why a box over seven free parameters is wasteful.
    """
    a = anisotropy()
    assert a["condition_number"] > 20.0
    assert a["axes_for_90pct_variance"] <= 4


def test_local_variation_breaks_perfect_correlation():
    """Two parameters sharing one cause must not come out at exactly +-1.

    Without local variation eta and vsat, both driven only by L, came out at
    -1.000. No real population shows that, and a corner model built on it
    would collapse a whole direction.
    """
    pop = sample_population(8000, seed=13)
    c = pop.correlation()
    off = c[~np.eye(len(PARAM_NAMES), dtype=bool)]
    assert np.max(np.abs(off)) < 0.97

    without = sample_population(8000, seed=13, local=False).correlation()
    i, j = PARAM_NAMES.index("eta"), PARAM_NAMES.index("vsat")
    assert abs(without[i, j]) > 0.99 > abs(c[i, j])


def test_vth0_mu0_process_correlation_is_negative():
    """Thinner oxide lowers vth0 and raises the fitted mu0: opposite signs.

    This is the claim the report leads with, and it is the one most likely to
    be broken by an edit to the sensitivity table.
    """
    pop = sample_population(8000, seed=4)
    c = pop.correlation()
    r = c[PARAM_NAMES.index("vth0"), PARAM_NAMES.index("mu0")]
    assert r < -0.15, f"expected a negative process correlation, got {r:+.3f}"


def test_process_and_estimation_correlation_disagree_in_sign():
    """The two matrices are different objects; the tests say so out loud."""
    from corners.deconvolve import design_from_stage1
    pop = sample_population(6000, seed=5)
    design = design_from_stage1(NOMINAL, NOISE)
    est_cov = extraction_covariance(NOMINAL, design.vgs, design.vds, NOISE)
    sd = np.sqrt(np.diag(est_cov))
    est = est_cov / np.outer(sd, sd)

    i, j = PARAM_NAMES.index("vth0"), PARAM_NAMES.index("mu0")
    assert pop.correlation()[i, j] < 0 < est[i, j]


def test_sampled_devices_respect_the_thermal_floor():
    pop = sample_population(5000, seed=6)
    assert np.all(pop.params[:, PARAM_NAMES.index("ss")] >= SS_FLOOR)


def test_every_parameter_has_at_least_one_physical_cause():
    caused = {p for v in SENSITIVITY.values() for p in v}
    assert set(PARAM_NAMES) <= caused


def test_sensitivity_table_only_names_real_parameters():
    for var, entry in SENSITIVITY.items():
        unknown = set(entry) - set(PARAM_NAMES)
        assert not unknown, f"{var} references {unknown}"
    known = {v.name for v in default_variables()}
    assert set(SENSITIVITY) <= known


# --------------------------------------------------------------- deconvolve

def test_project_psd_removes_negative_eigenvalues():
    A = np.diag([1.0, 0.5, -0.2])
    P, moved = project_psd(A)
    assert np.all(np.linalg.eigvalsh(P) >= -1e-12)
    assert moved > 0


def test_project_psd_leaves_a_psd_matrix_alone():
    A = np.diag([1.0, 0.5, 0.2])
    P, moved = project_psd(A)
    assert np.allclose(P, A) and moved == pytest.approx(0.0, abs=1e-12)


def test_deconvolution_recovers_a_known_spread():
    """Add a known 'extraction' covariance, take it back out, check the result."""
    pop = sample_population(6000, seed=8)
    nom = NOMINAL.to_dict()
    truth_cov = pop.covariance()

    rng = np.random.default_rng(0)
    added = np.diag([0.004 ** 2] * len(PARAM_NAMES))
    noise_draw = rng.standard_normal((len(pop), len(PARAM_NAMES))) * 0.004
    observed = pop.params * np.exp(noise_draw)

    dec = deconvolve(observed, nom, added)
    t = np.sqrt(np.diag(truth_cov))
    p = np.sqrt(np.clip(np.diag(dec.process), 0, None))
    for i, n in enumerate(PARAM_NAMES):
        assert 0.8 < p[i] / t[i] < 1.25, (
            f"{n}: recovered {p[i] * 100:.2f}% against truth {t[i] * 100:.2f}%")


def test_observed_spread_is_never_smaller_than_the_process_spread():
    """Independent noise adds variance; a negative diagonal means a bug."""
    pop = sample_population(3000, seed=9)
    rng = np.random.default_rng(1)
    observed = pop.params * np.exp(
        rng.standard_normal((len(pop), len(PARAM_NAMES))) * 0.01)
    obs = np.cov(np.log(observed / np.array([NOMINAL.to_dict()[n]
                                             for n in PARAM_NAMES])),
                 rowvar=False)
    assert np.all(np.diag(obs) >= np.diag(pop.covariance()) * 0.9)


# ------------------------------------------------------------------- corners

@pytest.fixture(scope="module")
def population():
    return sample_population(2000, seed=11)


def test_box_has_two_to_the_n_corners(population):
    cs = independent_box(population.covariance(), NOMINAL.to_dict(), 3.0)
    assert len(cs) == 2 ** len(PARAM_NAMES)


def test_box_reaches_outside_the_ellipsoid(population):
    """Why it over-covers: a box corner is much further than k sigma.

    In Mahalanobis distance the corner of a k-sigma box sits near
    k*sqrt(n) for independent parameters, and further when they correlate.
    """
    cov = population.covariance()
    pinv = np.linalg.pinv(cov, rcond=1e-8)
    cs = independent_box(cov, NOMINAL.to_dict(), 3.0)
    d = np.array([np.sqrt(x @ pinv @ x) for x in cs.log_offsets])
    assert d.max() > 3.0 * np.sqrt(len(PARAM_NAMES)) * 0.8


def test_pca_corners_sit_on_the_ellipsoid(population):
    """Each is exactly k sigma away, which is the point of using the axes.

    Measured with a pseudo-inverse, because the covariance is rank deficient
    and a plain inverse blows up along the direction the process cannot move
    in.
    """
    cov = population.covariance()
    pinv = np.linalg.pinv(cov, rcond=1e-8)
    cs = pca_corners(cov, NOMINAL.to_dict(), 3.0)
    for x in cs.log_offsets:
        assert np.sqrt(x @ pinv @ x) == pytest.approx(3.0, rel=0.05)


def test_pca_corners_concentrate_the_variance(population):
    """A few axes carry most of it, which is the whole argument for PCA."""
    cs = pca_corners(population.covariance(), NOMINAL.to_dict(), 3.0)
    assert any("variance" in n for n in cs.notes)
    w = np.sort(np.linalg.eigvalsh(population.covariance()))[::-1]
    assert float(np.sum(w[:3]) / np.sum(w)) > 0.85


def test_one_at_a_time_under_covers(population):
    """A sensitivity sweep is not a bound, and the report says so."""
    cs = one_at_a_time(population.covariance(), NOMINAL.to_dict(), 3.0)
    r = evaluate(cs, population.as_devices(), spec("ion"))
    assert r.coverage < r.nominal_coverage


def test_box_covers_but_wastes(population):
    cs = independent_box(population.covariance(), NOMINAL.to_dict(), 3.0)
    r = evaluate(cs, population.as_devices(), spec("ion"))
    assert r.safe
    assert r.waste > 0.25, "the box should be well wider than necessary"


def test_worst_case_distance_is_tighter_than_the_box(population):
    """Two corners against 128, and the comparison is the stage's result."""
    cov, nom = population.covariance(), NOMINAL.to_dict()
    devices = population.as_devices()
    m = spec("ion")
    box = evaluate(independent_box(cov, nom, 3.0), devices, m)
    wcd = evaluate(worst_case_distance(cov, nom, m.fn, 3.0), devices, m)
    assert wcd.safe
    assert wcd.waste < box.waste
    assert len(worst_case_distance(cov, nom, m.fn, 3.0)) == 2


def test_more_mc_samples_do_not_reduce_coverage(population):
    """Its coverage is a property of the sample size, not of the method."""
    cov, nom = population.covariance(), NOMINAL.to_dict()
    devices = population.as_devices()
    small = evaluate(statistical_mc(cov, nom, 50, 0), devices, spec("ion"))
    large = evaluate(statistical_mc(cov, nom, 2000, 0), devices, spec("ion"))
    assert large.coverage >= small.coverage


# ------------------------------------------------------------------ metrics

def test_metric_batch_matches_the_single_device_functions():
    """The batched evaluator is an optimisation; it must not change a number."""
    devs = sample_population(12, seed=12).as_devices()
    batch = evaluate_all(devs)
    for name, s in all_specs().items():
        one = np.array([s.fn(d) for d in devs])
        # Not bit-exact: the batch path asks for both bias points in one call,
        # so the floating-point ordering differs by ~1e-12 relative. A real
        # change in behaviour would be orders of magnitude larger.
        assert np.allclose(batch[name], one, rtol=1e-9), name


def test_delay_rises_when_drive_current_falls():
    fast = DeviceParams(mu0=0.036)
    slow = DeviceParams(mu0=0.024)
    assert spec("delay").fn(slow) > spec("delay").fn(fast)


def test_reference_range_widens_with_k():
    v = np.random.default_rng(0).normal(size=20000)
    lo2, hi2 = reference_range(v, 2.0)
    lo3, hi3 = reference_range(v, 3.0)
    assert lo3 < lo2 and hi3 > hi2
    assert nominal_coverage(3.0) > nominal_coverage(2.0)
