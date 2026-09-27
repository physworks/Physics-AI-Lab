"""Tests for the guarantees that matter.

Runnable either way:

    python tests/test_pipeline.py      # no dependencies
    pytest tests/test_pipeline.py      # if pytest is installed

The emphasis is on the invariants that keep the project honest -- whitelist
enforcement, the gate, provenance separation -- rather than on numerical
outputs, which are allowed to move as the models improve.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from wafer2device.contracts import Hypothesis, Provenance, VariationTerm
from wafer2device.gate import load_yaml
from wafer2device.orchestrator import Pipeline
from wafer2device.physics_checks.checks import run_all
from wafer2device.s1_pattern import descriptors as s1
from wafer2device.s1_pattern.data import generate_case, generate_dataset
from wafer2device.s2_hypothesis import rules
from wafer2device.s2_hypothesis.schema import (Whitelist, coerce_terms,
                                               validate_hypothesis)
from wafer2device.s3_device.models import AnalyticDeviceModel, variation_fields

PIPE_CFG = load_yaml(str(ROOT / "configs/pipeline.yaml"))
CHECKS_CFG = load_yaml(str(ROOT / "configs/checks.yaml"))
WL = Whitelist.from_config(load_yaml(str(ROOT / "configs/whitelist.yaml")))


def test_descriptors_are_finite_and_bounded():
    for case in generate_dataset(n_per_scenario=1):
        d = s1.extract_descriptors(case.wafer_map.value)
        assert 0.0 <= d.defect_rate <= 1.0
        assert 0.0 <= d.angular_anisotropy <= 1.0
        assert np.isfinite(d.radial_slope)
        assert len(d.radial_profile) == s1.N_RADIAL_BINS


def test_whitelist_rejects_unknown_parameter():
    h = Hypothesis(terms=[VariationTerm("plasma_mood", "edge_ring", 0.05)])
    ok, errors = validate_hypothesis(h, WL)
    assert not ok and "not whitelisted" in errors[0]


def test_whitelist_rejects_out_of_range_magnitude():
    h = Hypothesis(terms=[VariationTerm("gate_oxide_thickness", "edge_ring", 9.0)])
    ok, errors = validate_hypothesis(h, WL)
    assert not ok


def test_coercion_drops_invalid_terms_instead_of_repairing():
    raw = [{"param": "gate_oxide_thickness", "spatial_form": "edge_ring",
            "magnitude": 0.05},
           {"param": "not_a_real_param", "spatial_form": "edge_ring",
            "magnitude": 0.05},
           {"param": "gate_length", "spatial_form": "teleport", "magnitude": 0.01}]
    terms = coerce_terms(raw, WL)
    assert len(terms) == 1 and terms[0].param == "gate_oxide_thickness"


def test_analytic_model_returns_to_nominal_at_zero_variation():
    m = AnalyticDeviceModel()
    out = m.evaluate({"gate_oxide_thickness": np.zeros(5)})
    assert np.allclose(out["vth"], 0.45, atol=1e-9)


def test_vth_monotonic_in_oxide_thickness_and_doping():
    m = AnalyticDeviceModel()
    for param in ("gate_oxide_thickness", "channel_doping"):
        sweep = np.linspace(-0.08, 0.08, 21)
        vth = np.array([m.evaluate({param: np.array([x])})["vth"][0] for x in sweep])
        assert np.all(np.diff(vth) > 0), f"Vth not monotonic in {param}"


def test_wrong_spatial_form_fails_pattern_consistency():
    """A physically legal but spatially wrong hypothesis must be caught."""
    case = generate_case("t_edge", "edge_ring", n=41,
                         rng=np.random.default_rng(1))
    wrong = Hypothesis(terms=[VariationTerm("channel_doping", "center_spot", 0.10)])
    report = run_all(wrong, AnalyticDeviceModel(), case.wafer_map.value, CHECKS_CFG)
    pc = [r for r in report.results if r.name == "pattern_consistency"][0]
    assert not pc.passed and pc.direction_hint


def test_correct_hypothesis_passes_pattern_consistency():
    case = generate_case("t_edge2", "edge_ring", n=41,
                         rng=np.random.default_rng(2))
    right = Hypothesis(terms=[VariationTerm("gate_oxide_thickness", "edge_ring", 0.06)])
    report = run_all(right, AnalyticDeviceModel(), case.wafer_map.value, CHECKS_CFG)
    pc = [r for r in report.results if r.name == "pattern_consistency"][0]
    assert pc.passed


def test_strict_gate_blocks_downstream_when_checks_fail():
    """The gate is not advisory: a failed case must not produce a compensation."""
    cfg = load_yaml(str(ROOT / "configs/pipeline.yaml"))
    cfg["approval"]["mode"] = "strict"
    cfg["hypothesis"]["mode"] = "rule"
    pipe = Pipeline(cfg, CHECKS_CFG, load_yaml(str(ROOT / "configs/whitelist.yaml")))
    case = generate_case("t_comp", "edge_ring_plus_center", n=41,
                         rng=np.random.default_rng(3))
    pipe.run_case(case)
    if case.escalated:
        assert case.compensation is None
        assert case.device_response is None


def test_replan_loop_respects_max_rounds():
    cfg = load_yaml(str(ROOT / "configs/pipeline.yaml"))
    cfg["approval"]["mode"] = "auto"
    pipe = Pipeline(cfg, CHECKS_CFG, load_yaml(str(ROOT / "configs/whitelist.yaml")))
    case = generate_case("t_rounds", "edge_ring_plus_center", n=41,
                         rng=np.random.default_rng(4))
    pipe.run_case(case)
    assert len(case.replan_log) <= CHECKS_CFG["replan"]["max_rounds"]
    assert case.replan_log[-1].stop_reason is not None


def test_provenance_tags_are_never_mixed():
    cfg = load_yaml(str(ROOT / "configs/pipeline.yaml"))
    cfg["approval"]["mode"] = "auto"
    pipe = Pipeline(cfg, CHECKS_CFG, load_yaml(str(ROOT / "configs/whitelist.yaml")))
    case = generate_case("t_prov", "edge_ring", n=41, rng=np.random.default_rng(5))
    pipe.run_case(case)
    assert case.wafer_map.provenance is Provenance.SYNTHETIC
    assert case.hypothesis.provenance is Provenance.ASSUMED
    assert case.device_response.provenance is Provenance.PREDICTED
    assert case.compensation.provenance is Provenance.OPTIMIZED


def test_compensation_never_worsens_the_baseline():
    cfg = load_yaml(str(ROOT / "configs/pipeline.yaml"))
    cfg["approval"]["mode"] = "auto"
    pipe = Pipeline(cfg, CHECKS_CFG, load_yaml(str(ROOT / "configs/whitelist.yaml")))
    case = generate_case("t_opt", "radial_gradient", n=41,
                         rng=np.random.default_rng(6))
    pipe.run_case(case)
    c = case.compensation.value
    assert c.optimized_out_of_spec <= c.baseline_out_of_spec + 1e-9


def test_rule_path_is_deterministic():
    a = generate_case("t_det", "edge_ring", n=41, rng=np.random.default_rng(7))
    b = generate_case("t_det", "edge_ring", n=41, rng=np.random.default_rng(7))
    s1.run(a), s1.run(b)
    ha = rules.propose(a.descriptors.value, a.pattern.value, WL)
    hb = rules.propose(b.descriptors.value, b.pattern.value, WL)
    assert ha.signature() == hb.signature()


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}: {exc}")
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
