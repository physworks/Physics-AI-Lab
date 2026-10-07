"""The report generator, which writes the claims a reader actually sees.

This file exists because the generator shipped a sentence that was not
computed from anything: it said the LLM arm "matched the plain rule's answers
exactly" as a hardcoded string, and on a run where it did not match, the
report said it did.  A generator that states conclusions has to be tested like
any other code that states conclusions.

It also pins the underpowered guard.  Three repeats with a majority vote makes
the arm's accuracy a random variable -- two runs of the identical protocol
scored 87.5% and 83.3% -- and the guard is what stops that number being quoted
as if it were stable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_diagnose import _append_report

ARM = "llm:test-model"
CLASSES = ["none", "c1_kink", "negative_gds", "truncated_inner_loop",
           "bad_initial_guess", "singular_operating_point"]
MARKER = "## 5. Diagnosing convergence failures"


def _arm(name, acc_all, acc_dec, abst=0.0, abst_und=0):
    return {"arm": name, "n": 24, "accuracy_all": acc_all,
            "n_answered": 24, "accuracy_when_answered": acc_all,
            "abstention_rate": abst, "accuracy_on_decidable": acc_dec,
            "abstained_on_undecidable": abst_und, "n_undecidable": 3,
            "confusion": {}}


FRAGILITY = {
    "classes": [
        {"class": "negative_gds", "n": 4, "n_split": 0, "split_rate": 0.0,
         "mean_flip_fields": 1.0, "mean_evidence_chars": 862.0,
         "mean_history_length": 25.0},
        {"class": "truncated_inner_loop", "n": 4, "n_split": 2,
         "split_rate": 0.5, "mean_flip_fields": 3.0,
         "mean_evidence_chars": 569.0, "mean_history_length": 7.8},
        {"class": "bad_initial_guess", "n": 4, "n_split": 0,
         "split_rate": 0.0, "mean_flip_fields": 3.0,
         "mean_evidence_chars": 904.0, "mean_history_length": 25.0},
        {"class": "none", "n": 4, "n_split": 0, "split_rate": 0.0,
         "mean_flip_fields": 4.0, "mean_evidence_chars": 558.0,
         "mean_history_length": 7.2},
        {"class": "c1_kink", "n": 4, "n_split": 0, "split_rate": 0.0,
         "mean_flip_fields": 4.0, "mean_evidence_chars": 637.0,
         "mean_history_length": 12.2},
        {"class": "singular_operating_point", "n": 4, "n_split": 3,
         "split_rate": 0.75, "mean_flip_fields": 5.0,
         "mean_evidence_chars": 907.0, "mean_history_length": 25.0},
    ],
    "predictors": {"flip_count": 0.657, "history_length": 0.086,
                   "evidence_chars": 0.086},
    "concentration": {
        "split_cases": ["trunc_0", "trunc_2", "singular_0", "singular_1",
                        "singular_3"],
        "classes_hit": ["singular_operating_point", "truncated_inner_loop"],
        "cases_in_those_classes": 8, "p_exact": 0.0013175,
        "p_corrected": 0.019763, "n_class_groups": 15,
        "undecidable_overlap": [], "p_miss_undecidable": 0.47875},
    "flip_fields": {},
}


def _result(*, llm_answers=None, stability=None, reached=True,
            fragility=None):
    """A minimal results dict shaped like the real one."""
    truths = (["none"] * 4 + ["c1_kink"] * 4 + ["negative_gds"] * 4
              + ["truncated_inner_loop"] * 4 + ["bad_initial_guess"] * 4
              + ["singular_operating_point"] * 4)
    names = ([f"none_{i}" for i in range(4)] + [f"kink_{i}" for i in range(4)]
             + [f"neggds_{i}" for i in range(4)]
             + [f"trunc_{i}" for i in range(4)]
             + [f"badstart_{i}" for i in range(4)]
             + [f"singular_{i}" for i in range(4)])
    rule = {n: t for n, t in zip(names, truths)}
    for n in ("kink_1", "kink_2", "kink_3"):
        rule[n] = "none"
    llm = dict(rule)
    llm.update(llm_answers or {})

    per_case = [{"case": n, "truth": t, "rule": rule[n],
                 "rule_with_abstain": rule[n], ARM: llm[n]}
                for n, t in zip(names, truths)]
    right = sum(1 for r in per_case if r[ARM] == r["truth"])
    dec = [r for r in per_case if r["case"] not in
           ("kink_1", "kink_2", "kink_3")]
    right_dec = sum(1 for r in dec if r[ARM] == r["truth"])

    return {
        "n_cases": 24,
        "classes": CLASSES,
        "undecidable_from_trace": ["kink_1", "kink_2", "kink_3"],
        "llm_reached_model": reached,
        "model": "test-model",
        "stability": stability or {"repeats": 9, "cases_with_split_votes": 0,
                                   "split_cases": [],
                                   "split_on_undecidable": [],
                                   "votes": {}, "abstentions_used": 0},
        "fragility": fragility,
        "arms": {"rule": _arm("rule", 0.875, 1.0),
                 "rule_with_abstain": _arm("rule_with_abstain", 0.708, 0.810,
                                           abst=0.2917, abst_und=3),
                 ARM: _arm(ARM, right / 24, right_dec / len(dec))},
        "per_case": per_case,
    }


def _render(res, tmp_path: Path) -> str:
    (tmp_path / "report.md").write_text(
        f"# r\n\n{MARKER}\n\nstale content\n", encoding="utf-8")
    _append_report(res, tmp_path)
    return (tmp_path / "report.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# The verdict is computed, not asserted
# --------------------------------------------------------------------------

def test_matching_arm_is_described_as_matching(tmp_path):
    out = _render(_result(), tmp_path)
    assert "matched the plain rule's answers exactly" in out


def test_differing_arm_is_not_described_as_matching(tmp_path):
    """The bug: this sentence used to be a constant.

    One case answered differently, and the old generator still reported an
    exact match.
    """
    res = _result(llm_answers={"singular_2": "bad_initial_guess"})
    out = _render(res, tmp_path)
    assert "matched the plain rule's answers exactly" not in out
    assert "differed from the plain rule on" in out
    assert "`singular_2`" in out


def test_falling_below_the_rule_is_stated(tmp_path):
    res = _result(llm_answers={"singular_2": "bad_initial_guess"})
    out = _render(res, tmp_path)
    assert "fell below the rule" in out
    assert "95.2%" in out and "100.0%" in out


def test_tying_the_rule_does_not_claim_a_shortfall(tmp_path):
    out = _render(_result(), tmp_path)
    assert "fell below the rule" not in out


# --------------------------------------------------------------------------
# The split-vote branches
# --------------------------------------------------------------------------

def test_no_overlap_branch_states_the_probability(tmp_path):
    """0.66 under independence -- the reason no anti-correlation is claimed."""
    res = _result(stability={
        "repeats": 3, "cases_with_split_votes": 3,
        "split_cases": ["trunc_1", "singular_2", "singular_3"],
        "split_on_undecidable": [], "votes": {}, "abstentions_used": 0})
    out = _render(res, tmp_path)
    assert "None of those splits fall on the undecidable cases" in out
    assert "0.66" in out
    assert "says nothing about which cases the trace cannot decide" in out


def test_overlap_branch_says_instability_replaced_abstention(tmp_path):
    res = _result(stability={
        "repeats": 3, "cases_with_split_votes": 2,
        "split_cases": ["kink_1", "trunc_1"],
        "split_on_undecidable": ["kink_1"], "votes": {},
        "abstentions_used": 0})
    out = _render(res, tmp_path)
    assert "fall on the undecidable cases" in out
    assert "0.66" not in out, "the probability argument belongs to the other branch"


def test_the_two_split_branches_are_exclusive(tmp_path):
    res = _result(stability={
        "repeats": 9, "cases_with_split_votes": 1,
        "split_cases": ["kink_2"], "split_on_undecidable": ["kink_2"],
        "votes": {}, "abstentions_used": 0})
    out = _render(res, tmp_path)
    assert "None of those splits" not in out


def test_unanimous_run_claims_neither_branch(tmp_path):
    out = _render(_result(), tmp_path)
    assert "None of those splits" not in out
    assert "fall on the undecidable cases" not in out


# --------------------------------------------------------------------------
# The underpowered guard
# --------------------------------------------------------------------------

def test_three_repeats_with_splits_is_flagged_underpowered(tmp_path):
    res = _result(stability={
        "repeats": 3, "cases_with_split_votes": 3,
        "split_cases": ["trunc_1", "singular_2", "singular_3"],
        "split_on_undecidable": [], "votes": {}, "abstentions_used": 0})
    out = _render(res, tmp_path)
    assert "UNDERPOWERED" in out
    assert "--repeats 9" in out


def test_nine_repeats_is_not_flagged(tmp_path):
    res = _result(stability={
        "repeats": 9, "cases_with_split_votes": 2,
        "split_cases": ["trunc_1", "singular_3"],
        "split_on_undecidable": [], "votes": {}, "abstentions_used": 0})
    out = _render(res, tmp_path)
    assert "UNDERPOWERED" not in out


def test_a_stable_run_is_not_flagged(tmp_path):
    out = _render(_result(), tmp_path)
    assert "UNDERPOWERED" not in out


# --------------------------------------------------------------------------
# Fallback runs are not evidence
# --------------------------------------------------------------------------

def test_fallback_run_is_labelled_and_makes_no_model_claim(tmp_path):
    out = _render(_result(reached=False), tmp_path)
    assert "rule fallback" in out
    assert "not evidence about a model" in out
    assert "matched the plain rule's answers exactly" not in out
    assert "UNDERPOWERED" not in out


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------

def test_stale_section_is_replaced_not_appended(tmp_path):
    out = _render(_result(), tmp_path)
    assert out.count(MARKER) == 1
    assert "stale content" not in out


def test_every_arm_gets_a_table_row(tmp_path):
    res = _result()
    out = _render(res, tmp_path)
    for name in res["arms"]:
        assert f"`{name}`" in out


# --------------------------------------------------------------------------
# The concentration section
# --------------------------------------------------------------------------

def _split_result(**kw):
    return _result(stability={
        "repeats": 9, "cases_with_split_votes": 5,
        "split_cases": FRAGILITY["concentration"]["split_cases"],
        "split_on_undecidable": [], "votes": {}, "abstentions_used": 0},
        **kw)


def test_concentration_section_states_both_p_values(tmp_path):
    out = _render(_split_result(fragility=FRAGILITY), tmp_path)
    assert "Where the splits fall" in out
    assert "0.0013" in out, "the exact probability"
    assert "0.020" in out, "the corrected probability"
    assert "15 groups of classes" in out


def test_concentration_section_names_the_two_classes(tmp_path):
    out = _render(_split_result(fragility=FRAGILITY), tmp_path)
    assert "`singular_operating_point`" in out
    assert "`truncated_inner_loop`" in out
    assert "2 of 6 classes" in out


def test_every_class_gets_a_fragility_row(tmp_path):
    out = _render(_split_result(fragility=FRAGILITY), tmp_path)
    for row in FRAGILITY["classes"]:
        assert f"| `{row['class']}` | {row['n_split']}/{row['n']} |" in out


def test_the_cheap_explanation_is_reported_as_ruled_out(tmp_path):
    out = _render(_split_result(fragility=FRAGILITY), tmp_path)
    assert "+0.09" in out, "the near-zero confound correlations"
    assert "+0.66" in out, "the flip-count correlation"
    assert "histories as long as the two that do" in out


def test_the_correlation_is_not_presented_as_a_mechanism(tmp_path):
    """The caveat is load-bearing: six classes of four cases is not a result."""
    out = _render(_split_result(fragility=FRAGILITY), tmp_path)
    assert "direction to test, not a mechanism" in out
    assert "known" in out and "exception" in out


def test_no_concentration_section_without_fragility_data(tmp_path):
    out = _render(_split_result(fragility=None), tmp_path)
    assert "Where the splits fall" not in out


def test_no_concentration_section_when_nothing_split(tmp_path):
    out = _render(_result(fragility=FRAGILITY), tmp_path)
    assert "Where the splits fall" not in out


def test_undecidable_paragraph_points_forward_instead_of_contradicting(
        tmp_path):
    """The two sections must not read as disagreeing.

    One says the splits carry no information about which cases the trace
    cannot decide; the other says they do track class entanglement.  Both are
    true and the wording has to keep them apart.
    """
    out = _render(_split_result(fragility=FRAGILITY), tmp_path)
    assert "says nothing about which cases the trace cannot decide" in out
    assert "carries no information about which cases are hard" not in out


def test_abstention_count_comes_from_the_data(tmp_path):
    res = _result(stability={
        "repeats": 9, "cases_with_split_votes": 0, "split_cases": [],
        "split_on_undecidable": [], "votes": {}, "abstentions_used": 5})
    out = _render(res, tmp_path)
    assert "abstention 5 times" in out
