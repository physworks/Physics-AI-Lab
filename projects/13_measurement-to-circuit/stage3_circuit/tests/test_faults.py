"""Axis 3 -- the fault answer key and the rule diagnosers.

Every accuracy number this stage reports is relative to this answer key, so
the key's own properties have to hold: balanced classes, a fault actually
injected, a trace that carries symptoms rather than labels, and an undecidable
set that is derived from the evidence instead of from the labels.

These tests share one session-scoped solve of all 24 cases (see conftest);
solving them costs about 40 seconds and no test mutates a trace.
"""

from __future__ import annotations

import numpy as np
import pytest

from circuit.device import DeviceParams
from circuit.diagnose import ABSTAIN, rule_diagnose
from circuit.faults import FAULT_CLASSES, build_cases, solve_case

from run_diagnose import undecidable_cases

EXPECTED_PER_CLASS = 4


# --------------------------------------------------------------------------
# The answer key
# --------------------------------------------------------------------------

def test_case_set_is_balanced(cases):
    counts = {c: 0 for c in FAULT_CLASSES}
    for case in cases:
        counts[case.truth] += 1
    assert len(cases) == len(FAULT_CLASSES) * EXPECTED_PER_CLASS
    assert set(counts.values()) == {EXPECTED_PER_CLASS}, counts


def test_case_names_are_unique(cases):
    names = [c.name for c in cases]
    assert len(set(names)) == len(names)


def test_case_truths_are_known_classes(cases):
    assert all(c.truth in FAULT_CLASSES for c in cases)


def test_build_cases_is_reproducible():
    """The random far-starts are seeded, so the key must not drift."""
    a = {c.name: c.truth for c in build_cases(seed=0)}
    b = {c.name: c.truth for c in build_cases(seed=0)}
    assert a == b
    x_a = [c.x0 for c in build_cases(seed=0) if c.x0 is not None]
    x_b = [c.x0 for c in build_cases(seed=0) if c.x0 is not None]
    assert all(np.allclose(p, q) for p, q in zip(x_a, x_b))


def test_no_trace_contains_its_own_label(traces, truths):
    """Mechanical check across all 24, keys and values."""
    for name, t in traces.items():
        flat = str(t)
        for cls in FAULT_CLASSES:
            if cls == "none":
                continue
            assert cls not in flat, f"{name} leaks the label {cls!r}"


# --------------------------------------------------------------------------
# The faults are actually injected
# --------------------------------------------------------------------------

def test_healthy_cases_converge(traces, truths):
    for name, t in traces.items():
        if truths[name] == "none":
            assert t["converged"], f"{name} should have been a clean solve"


def test_negative_gds_cases_report_negative_conductance(traces, truths):
    for name, t in traces.items():
        if truths[name] == "negative_gds":
            d = (t["device_state"] or [{}])[0]
            assert float(d.get("gds", 0.0)) < 0.0, name


def test_truncated_cases_report_an_unconverged_inner_loop(traces, truths):
    for name, t in traces.items():
        if truths[name] == "truncated_inner_loop":
            d = (t["device_state"] or [{}])[0]
            assert not bool(d.get("inner_converged", True)), name


def test_bad_start_cases_recover_from_the_default_start(traces, truths):
    """The class is defined by this asymmetry; without it there is no signal."""
    for name, t in traces.items():
        if truths[name] == "bad_initial_guess":
            assert not t["converged"], name
            assert t["retry_from_default_converged"], name
            assert t["started_far_from_default"], name


def test_singular_cases_fail_from_both_starts(traces, truths):
    """No solution exists, so the retry must not rescue them.

    This is what separates ``singular_operating_point`` from
    ``bad_initial_guess``: both fail, only one recovers.
    """
    for name, t in traces.items():
        if truths[name] == "singular_operating_point":
            assert not t["converged"], name
            assert not t["retry_from_default_converged"], name


def test_kink_cases_are_the_invisible_ones(traces, truths):
    """The finding: a discontinuity the solve never crossed leaves no trace.

    Three of the four kink cases converge cleanly, which is precisely why
    they cannot be told from healthy. If this ever changes, the undecidable
    count in the README changes with it.
    """
    clean = [n for n, t in traces.items()
             if truths[n] == "c1_kink" and t["converged"]]
    assert len(clean) == 3, clean


# --------------------------------------------------------------------------
# Undecidability is derived from the evidence, not from the labels
# --------------------------------------------------------------------------

def test_undecidable_set_is_the_three_clean_kinks(traces, truths):
    und = undecidable_cases(traces, truths)
    assert sorted(und) == ["kink_1", "kink_2", "kink_3"], und


def test_undecidable_traces_are_indistinguishable_from_healthy(traces, truths):
    """The claim behind reporting accuracy on a subset, checked field by field.

    Every field the rule consults has the same value on an undecidable case
    as on a healthy one.  No rule can separate them, so neither arm is
    charged for failing to.
    """
    fields = ("converged", "retry_from_default_converged")
    healthy = [t for n, t in traces.items() if truths[n] == "none"]
    for name in undecidable_cases(traces, truths):
        t = traces[name]
        d = (t["device_state"] or [{}])[0]
        assert float(d.get("gds", 0.0)) >= 0.0
        assert bool(d.get("inner_converged", True))
        assert float(d.get("gm", 1.0)) != 0.0
        for f in fields:
            assert any(t[f] == h[f] for h in healthy), (name, f)


def test_healthy_cases_are_never_called_undecidable(traces, truths):
    und = set(undecidable_cases(traces, truths))
    assert not any(truths[n] == "none" for n in und)


# --------------------------------------------------------------------------
# The rule arms
# --------------------------------------------------------------------------

def test_rule_is_deterministic(traces):
    for t in traces.values():
        assert rule_diagnose(t) == rule_diagnose(t)


def test_rule_answers_only_known_classes(traces):
    for t in traces.values():
        assert rule_diagnose(t) in FAULT_CLASSES
        assert rule_diagnose(t, allow_abstain=True) in FAULT_CLASSES + (ABSTAIN,)


def test_rule_is_perfect_on_the_decidable_subset(traces, truths):
    """The headline the LLM arm is compared against: 100%, 21 of 21."""
    und = set(undecidable_cases(traces, truths))
    decidable = [n for n in traces if n not in und]
    wrong = [n for n in decidable if rule_diagnose(traces[n]) != truths[n]]
    assert not wrong, wrong


def test_rule_overall_accuracy_is_twenty_one_of_twenty_four(traces, truths):
    right = sum(1 for n, t in traces.items() if rule_diagnose(t) == truths[n])
    assert right == 21
    assert np.isclose(right / len(traces), 0.875)


def test_rule_misses_exactly_the_undecidable_cases(traces, truths):
    und = set(undecidable_cases(traces, truths))
    wrong = {n for n, t in traces.items() if rule_diagnose(t) != truths[n]}
    assert wrong == und


def test_abstaining_rule_never_answers_wrongly(traces, truths):
    """Its whole value: when it does answer, it is right."""
    answered = [(n, rule_diagnose(t, allow_abstain=True))
                for n, t in traces.items()]
    for name, ans in answered:
        if ans != ABSTAIN:
            assert ans == truths[name], (name, ans, truths[name])


def test_abstaining_rule_catches_every_undecidable_case(traces, truths):
    und = set(undecidable_cases(traces, truths))
    for name in und:
        assert rule_diagnose(traces[name], allow_abstain=True) == ABSTAIN


def test_abstention_costs_the_healthy_cases(traces, truths):
    """The trade that makes the choice a cost judgement, not a tuning one.

    A healthy circuit and an unexcited defect produce the same trace, so any
    rule that abstains on one abstains on the other.  Three invisible faults
    caught, four healthy cases given up: 7 abstentions.
    """
    abst = [n for n, t in traces.items()
            if rule_diagnose(t, allow_abstain=True) == ABSTAIN]
    assert len(abst) == 7
    assert sum(1 for n in abst if truths[n] == "none") == 4


def test_no_rule_variant_can_separate_healthy_from_unexcited_fault(traces,
                                                                   truths):
    """The impossibility stated as a test rather than as prose.

    If some field did separate them, the two variants would not be the only
    options and the README's "quietly wrong three times or silent seven
    times" claim would be wrong.
    """
    und = undecidable_cases(traces, truths)
    healthy = [n for n, _ in traces.items() if truths[n] == "none"]
    for u in und:
        assert any(rule_diagnose(traces[u]) == rule_diagnose(traces[h])
                   for h in healthy)


# --------------------------------------------------------------------------
# Fault injection is what changes the answer
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fault,strength", [("negative_gds", 6.0),
                                            ("truncated_inner_loop", 1.0)])
def test_injected_fault_changes_the_device_state(fault, strength):
    """A fault that does not change the evidence is not being tested."""
    from circuit.faults import _inverter_with
    p = DeviceParams()
    clean = solve_case(type("C", (), {
        "name": "c", "truth": "none", "x0": None,
        "circuit": _inverter_with(p, 1.0, "none", 1.0)})())
    dirty = solve_case(type("C", (), {
        "name": "d", "truth": fault, "x0": None,
        "circuit": _inverter_with(p, 1.0, fault, strength)})())
    assert (clean["device_state"][0] != dirty["device_state"][0]
            or clean["converged"] != dirty["converged"])
