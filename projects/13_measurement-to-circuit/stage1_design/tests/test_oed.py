"""Tests for stage 1.

The ones that matter are the ones that would catch a wrong result rather than
a crash: that the cached Fisher arithmetic equals the direct computation, that
the noise model actually changes where points go, that the agent cannot widen
its own remit, and that a design violating a hard constraint is refused rather
than quietly accepted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oed import (DesignConstraints, DeviceParams, NoiseModel, build_pool,
                 decide, default_required_regions, greedy_design,
                 holdout_grid, information, run_arm, uniform_design,
                 validate, violations)
from oed.fisher import build_cache, score
from oed.objective import (ObjectiveRejected, full_objective, rule_objective,
                           rule_objective_provenance)
from oed.pipeline import run_stage, screen
from oed.scenarios import ALL_PARAMS, SUBJECT, WITHOUT_THETA, breakeven_sweep

TRUTH = DeviceParams()
NOISE = NoiseModel()


@pytest.fixture(scope="module")
def setup():
    cons = DesignConstraints(n_points=10, max_distinct_vds=3,
                             required_regions=default_required_regions())
    pool = build_pool(TRUTH, cons)
    cache = build_cache(TRUTH, pool.vgs, pool.vds, NOISE)
    return cons, pool, cache


# --------------------------------------------------------------------- noise

def test_noise_floor_dominates_at_low_current():
    n = NoiseModel(sigma_rel=0.01, i_floor=1e-12)
    assert n.relative(1e-4) == pytest.approx(0.01, rel=1e-3)
    assert n.relative(1e-12) > 1.0          # at the floor, nothing is resolved
    assert np.all(np.diff(n.relative(np.array([1e-12, 1e-10, 1e-8]))) < 0)


def test_uniform_noise_model_ignores_current():
    n = NoiseModel(floor_aware=False)
    assert n.relative(1e-12) == pytest.approx(n.relative(1e-3))


# -------------------------------------------------------------------- fisher

def test_cache_matches_direct(setup):
    _cons, pool, cache = setup
    idx = list(range(0, 60, 4))
    direct = information(TRUTH, pool.vgs[idx], pool.vds[idx], NOISE,
                         ["vth0", "ss", "mu0"]).matrix
    assert np.allclose(cache.subset(idx, ["vth0", "ss", "mu0"]), direct,
                       rtol=1e-10, atol=0)


def test_more_points_never_lose_information(setup):
    """Fisher information is additive, so adding a point cannot reduce it."""
    _cons, _pool, cache = setup
    small = cache.subset(list(range(0, 30, 3)), ALL_PARAMS)
    large = cache.subset(list(range(0, 60, 3)), ALL_PARAMS)
    assert np.all(np.linalg.eigvalsh(large - small) > -1e-9)


def test_criteria_agree_on_a_strictly_better_design(setup):
    _cons, _pool, cache = setup
    few, many = list(range(0, 24, 3)), list(range(0, 72, 3))
    for crit in ("D", "A", "E"):
        assert (score(cache.subset(many, ALL_PARAMS), crit)
                > score(cache.subset(few, ALL_PARAMS), crit))


def test_unknown_criterion_is_refused(setup):
    _cons, _pool, cache = setup
    with pytest.raises(ValueError):
        score(cache.subset([1, 2, 3], ALL_PARAMS), "Z")


# --------------------------------------------------------------- constraints

def test_pool_excludes_points_over_the_power_limit():
    cons = DesignConstraints(n_points=8, power_limit_w=1e-5,
                             required_regions=default_required_regions())
    pool = build_pool(TRUTH, cons)
    assert not np.all(pool.allowed)
    assert np.all(pool.current[pool.allowed] * pool.vds[pool.allowed]
                  <= cons.power_limit_w + 1e-18)


def test_design_respects_distinct_vds_cap(setup):
    cons, pool, cache = setup
    d = greedy_design(TRUTH, pool, cons, NOISE,
                      full_objective(TRUTH.to_dict()), cache=cache)
    assert pool.distinct_vds(d.indices) <= cons.max_distinct_vds
    assert d.violations == []


def test_missing_specification_point_is_a_violation(setup):
    cons, pool, _cache = setup
    idx = [i for i in pool.allowed_index().tolist()
           if 0.3 < pool.vgs[i] < 0.9][:cons.n_points]
    problems = violations(pool, idx, cons)
    assert any("off_state" in p or "on_state" in p for p in problems)


# ------------------------------------------------------------------ objective

def test_validate_rejects_unknown_parameter():
    with pytest.raises(ObjectiveRejected):
        validate({"include_params": ["vth0", "not_a_parameter"]},
                 ALL_PARAMS, TRUTH.to_dict())


def test_validate_rejects_extra_keys():
    with pytest.raises(ObjectiveRejected):
        validate({"include_params": ["vth0", "ss"], "n_points": 40},
                 ALL_PARAMS, TRUTH.to_dict())


def test_validate_rejects_empty_and_single_objective():
    for bad in ([], ["vth0"]):
        with pytest.raises(ObjectiveRejected):
            validate({"include_params": bad}, ALL_PARAMS, TRUTH.to_dict())


def test_validate_requires_c_target_inside_the_objective():
    with pytest.raises(ObjectiveRejected):
        validate({"include_params": ["vth0", "ss"], "criterion": "c",
                  "c_target": "mu0"}, ALL_PARAMS, TRUTH.to_dict())


def test_dropped_parameters_are_pinned_not_forgotten():
    obj = validate({"include_params": WITHOUT_THETA}, ALL_PARAMS,
                   TRUTH.to_dict())
    assert SUBJECT in obj.fixed_params
    assert obj.fixed_params[SUBJECT] == pytest.approx(TRUTH.theta)


# ---------------------------------------------------------------------- agent

def test_agent_falls_back_without_a_key(setup):
    cons, pool, _cache = setup
    ev = screen(TRUTH, pool, NOISE)
    dec = decide(ev, cons.describe(), TRUTH.to_dict(), "any purpose",
                 api_key="")
    assert not dec.used_llm
    assert dec.objective.source == "rule"


def test_agent_rejects_an_out_of_schema_answer(setup):
    """A model that tries to change the point count is refused, not obeyed."""
    cons, pool, _cache = setup
    ev = screen(TRUTH, pool, NOISE)
    dec = decide(ev, cons.describe(), TRUTH.to_dict(), "p",
                 transport=lambda _p: json.dumps(
                     {"include_params": ALL_PARAMS, "criterion": "D",
                      "n_points": 500}))
    assert not dec.used_llm
    assert any("unexpected key" in e for e in dec.errors)


def test_agent_accepts_a_well_formed_answer(setup):
    cons, pool, _cache = setup
    ev = screen(TRUTH, pool, NOISE)
    dec = decide(ev, cons.describe(), TRUTH.to_dict(), "p",
                 transport=lambda _p: json.dumps(
                     {"include_params": WITHOUT_THETA, "criterion": "A",
                      "rationale": "theta is undeterminable"}))
    assert dec.used_llm
    assert dec.objective.criterion == "A"
    assert SUBJECT not in dec.objective.include_params


def test_malformed_json_falls_back_rather_than_raising(setup):
    cons, pool, _cache = setup
    ev = screen(TRUTH, pool, NOISE)
    dec = decide(ev, cons.describe(), TRUTH.to_dict(), "p",
                 transport=lambda _p: "not json at all")
    assert not dec.used_llm and dec.objective.source == "rule"


# ------------------------------------------------------------------ providers

_RESPONSE = {
    "openai": {"choices": [{"message": {"content": '{"ok": 1}'}}]},
    "anthropic": {"content": [{"type": "text", "text": '{"ok": 1}'}]},
    "gemini": {"candidates": [{"content": {"parts": [{"text": '{"ok":'},
                                                     {"text": ' 1}'}]}}]},
}


def _capture(provider: str, model: str):
    """Build one request without touching the network."""
    import json as _json
    import unittest.mock as um
    from oed.agent import _call_model

    cap = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return _json.dumps(_RESPONSE[provider]).encode()

    def _fake(req, timeout=0):
        cap["url"] = req.full_url
        cap["headers"] = {k.lower(): v for k, v in req.headers.items()}
        cap["body"] = _json.loads(req.data.decode())
        return _Resp()

    with um.patch("urllib.request.urlopen", _fake):
        cap["text"] = _call_model({"x": 1}, model, "SECRET_KEY", provider)
    return cap


@pytest.mark.parametrize("model,provider", [
    ("gpt-4o-mini", "openai"), ("o3-mini", "openai"),
    ("claude-sonnet-4-5", "anthropic"),
    ("gemini-3.5-flash-lite", "gemini"), ("gemini-3.8-flash", "gemini"),
])
def test_provider_inferred_from_model_name(model, provider):
    from oed.agent import provider_for
    assert provider_for(model) == provider


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-4o-mini"), ("anthropic", "claude-sonnet-4-5"),
    ("gemini", "gemini-3.5-flash-lite"),
])
def test_every_provider_parses_back_to_the_same_json(provider, model):
    assert json.loads(_capture(provider, model)["text"]) == {"ok": 1}


@pytest.mark.parametrize("provider,model,header", [
    ("openai", "gpt-4o-mini", "authorization"),
    ("anthropic", "claude-sonnet-4-5", "x-api-key"),
    ("gemini", "gemini-3.5-flash-lite", "x-goog-api-key"),
])
def test_key_travels_in_a_header_never_in_the_url(provider, model, header):
    """Gemini's docs show ?key=; a key in a URL ends up in logs and history."""
    cap = _capture(provider, model)
    assert header in cap["headers"]
    assert "SECRET_KEY" not in cap["url"]


def test_v1_prompt_argues_only_one_side_of_the_trade():
    """Why the first measured run failed: the prompt never named the bias.

    Pinned as a regression, because the finding in the report depends on this
    being a property of v1 and not of the model that followed it.
    """
    from oed.agent import SYSTEM_PROMPT_V1, system_prompt
    assert "BIAS" not in SYSTEM_PROMPT_V1
    assert "nominal_provenance" not in SYSTEM_PROMPT_V1
    assert system_prompt(1) == SYSTEM_PROMPT_V1


def test_v2_prompt_states_both_sides_without_naming_the_boundary():
    from oed.agent import SYSTEM_PROMPT
    assert "BIAS" in SYSTEM_PROMPT and "VARIANCE" in SYSTEM_PROMPT
    assert "nominal_provenance" in SYSTEM_PROMPT
    for leak in ("20%", "20 percent", "30%"):
        assert leak not in SYSTEM_PROMPT, "the prompt must not name the answer"


def test_provenance_reaches_the_model():
    """The defect that made the first comparison unfair."""
    from oed.agent import build_request
    ev = {"rel_sigma": {"vth0": 0.001}, "sensitivity": {"vth0": 1.0},
          "strong_correlations": [], "condition_number": 1e5,
          "nominal_provenance": {
              "theta": {"source": "generic default", "rel_uncertainty": 0.45}}}
    payload = build_request(ev, {"n_points": 12}, "purpose")
    prov = payload["evidence"]["nominal_provenance"]
    assert prov["theta"]["rel_uncertainty_percent"] == 45.0


@pytest.mark.parametrize("version", [1, 2])
def test_prompt_version_is_recorded_on_the_decision(version):
    dec = decide({"rel_sigma": {}, "sensitivity": {},
                  "strong_correlations": [], "condition_number": 1.0},
                 {"n_points": 12}, TRUTH.to_dict(), "p",
                 prompt_version=version,
                 transport=lambda _p: json.dumps(
                     {"include_params": ["vth0", "ss"], "criterion": "D"}))
    assert dec.provenance()["prompt_version"] == version


def test_dotenv_reads_a_utf16_file(tmp_path):
    """PowerShell redirection writes UTF-16; the loader must not choke."""
    from oed.agent import load_dotenv
    (tmp_path / ".env").write_bytes("OED_UTF16=works\n".encode("utf-16"))
    load_dotenv(str(tmp_path / ".env"))
    assert __import__("os").environ.get("OED_UTF16") == "works"


def test_dotenv_does_not_override_a_real_environment_variable(tmp_path,
                                                             monkeypatch):
    from oed.agent import load_dotenv
    (tmp_path / ".env").write_text("OED_TEST_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("OED_TEST_KEY", "from_env")
    load_dotenv(str(tmp_path / ".env"))
    assert __import__("os").environ["OED_TEST_KEY"] == "from_env"


# ----------------------------------------------------------------- the rules

def test_sensitivity_rule_ignores_provenance(setup):
    """The blind spot the break-even sweep exposes, pinned as behaviour."""
    _cons, pool, _cache = setup
    decisions = set()
    for trust in (0.02, 0.60):
        case = [c for c in breakeven_sweep()
                if abs(c.rel_uncertainty - trust) < 1e-9][0]
        ev = screen(TRUTH, pool, NOISE, case.provenance())
        obj = rule_objective(ev, TRUTH.to_dict())
        decisions.add(SUBJECT in obj.include_params)
    assert len(decisions) == 1, "sensitivity rule should not react to trust"


def test_provenance_rule_reacts_to_trust(setup):
    _cons, pool, _cache = setup
    got = {}
    for trust in (0.02, 0.60):
        case = [c for c in breakeven_sweep()
                if abs(c.rel_uncertainty - trust) < 1e-9][0]
        ev = screen(TRUTH, pool, NOISE, case.provenance())
        obj = rule_objective_provenance(ev, TRUTH.to_dict())
        got[trust] = SUBJECT in obj.include_params
    assert got[0.02] is False and got[0.60] is True


# ------------------------------------------------------------------- headline

def test_optimal_design_beats_the_uniform_grid():
    """The result stage 1 exists to produce.  If this fails, nothing holds."""
    cons = DesignConstraints(n_points=12, max_distinct_vds=3,
                             required_regions=default_required_regions())
    pool = build_pool(TRUTH, cons)
    cache = build_cache(TRUTH, pool.vgs, pool.vds, NOISE)
    obj = full_objective(TRUTH.to_dict(), "D")
    hold = holdout_grid()

    uni = run_arm(TRUTH, uniform_design(TRUTH, pool, cons, NOISE, obj),
                  NOISE, seeds=8, holdout=hold)
    opt = run_arm(TRUTH, greedy_design(TRUTH, pool, cons, NOISE, obj,
                                       arm="greedy_D", cache=cache),
                  NOISE, seeds=8, holdout=hold)
    assert opt.predict_rms_log10 < uni.predict_rms_log10


def test_fisher_prediction_tracks_the_realised_spread():
    """A design chosen by a Fisher matrix is only as good as that matrix.

    Project 12 predicted the spread with a uniform-noise assumption and was
    1.54x optimistic on ``ss``.  With floor-aware weights the same check has
    to come back close, or the designs here are built on a wrong number.
    """
    cons = DesignConstraints(n_points=12, max_distinct_vds=3,
                             required_regions=default_required_regions())
    pool = build_pool(TRUTH, cons)
    cache = build_cache(TRUTH, pool.vgs, pool.vds, NOISE)
    d = greedy_design(TRUTH, pool, cons, NOISE,
                      full_objective(TRUTH.to_dict(), "D"), cache=cache)
    r = run_arm(TRUTH, d, NOISE, seeds=30)

    for n in ("vth0", "ss", "mu0", "eta"):
        pred, real = r.predicted_rel_sigma[n], r.realised_rel_sigma[n]
        assert 0.5 < real / pred < 2.0, (
            f"{n}: predicted {pred * 100:.3f}%, realised {real * 100:.3f}%")


def test_stage_escalates_rather_than_relaxing_a_constraint():
    """An impossible brief must stop, not quietly return a smaller design."""
    cons = DesignConstraints(n_points=40, max_distinct_vds=1,
                             required_regions=default_required_regions())
    pool = build_pool(TRUTH, cons)
    outcome = run_stage(TRUTH, pool, cons, NOISE, "p", arm="textbook")
    assert outcome.escalated and outcome.design is None
