"""The fragility measurement, which carries a claim the README quotes.

The nine-repeat run split on 5 of 24 cases, all of them inside two classes.
This module says how surprising that is and which case properties track it,
so its arithmetic has to be right -- a wrong p-value here would turn sampling
noise into a reported finding.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from circuit.fragility import (DEVICE_FIELDS, TOP_FIELDS, _p_all_inside,
                               _p_none_inside, _spearman, class_rows,
                               concentration, flip_fields, predictors,
                               summarise)

#: Splits observed in the two nine-repeat runs.  Both are kept because the
#: concentration replicated and the membership did not: every split in both
#: runs is inside the same two classes, but which cases inside them move
#: changes.  That is the difference between a class-level property and a
#: case-level one, and a single fixture would hide it.
SPLIT_9 = ["trunc_0", "trunc_2", "singular_0", "singular_1", "singular_3"]
SPLIT_9B = ["trunc_1", "trunc_2", "singular_0", "singular_1", "singular_2",
            "singular_3"]
UND = ["kink_1", "kink_2", "kink_3"]


# --------------------------------------------------------------------------
# The probability arithmetic
# --------------------------------------------------------------------------

def test_p_all_inside_matches_the_binomial_coefficients():
    assert np.isclose(_p_all_inside(5, 8, 24), comb(8, 5) / comb(24, 5))
    assert np.isclose(_p_all_inside(3, 8, 24), comb(8, 3) / comb(24, 3))


def test_p_none_inside_matches_the_binomial_coefficients():
    assert np.isclose(_p_none_inside(5, 3, 24), comb(21, 5) / comb(24, 5))
    assert np.isclose(_p_none_inside(3, 3, 24), comb(21, 3) / comb(24, 3))


def test_impossible_placements_have_probability_zero():
    assert _p_all_inside(9, 8, 24) == 0.0
    assert _p_none_inside(23, 3, 24) == 0.0


def test_a_whole_subset_is_certain():
    assert np.isclose(_p_all_inside(5, 24, 24), 1.0)
    assert np.isclose(_p_none_inside(5, 0, 24), 1.0)


def test_no_splits_gives_no_probability_rather_than_a_false_one():
    """Zero splits is not evidence of anything; it must not read as p=1."""
    assert np.isnan(_p_all_inside(0, 8, 24))
    assert np.isnan(_p_none_inside(0, 3, 24))


# --------------------------------------------------------------------------
# Concentration on the real answer key
# --------------------------------------------------------------------------

def test_concentration_finds_the_two_classes(truths):
    c = concentration(SPLIT_9, truths, UND)
    assert c.classes_hit == ["singular_operating_point",
                             "truncated_inner_loop"]
    assert c.cases_in_those_classes == 8


def test_concentration_p_values_are_the_quoted_ones(truths):
    c = concentration(SPLIT_9, truths, UND)
    assert np.isclose(c.p_exact, 0.0013175, atol=1e-6)
    assert np.isclose(c.p_corrected, 0.019763, atol=1e-5)
    assert c.n_class_pairs == comb(6, 2)


def test_the_second_run_replicates_the_concentration(truths):
    """Six splits, same two classes, and an order of magnitude more extreme.

    This is what lifts the finding above "one striking run": the corrected
    probability for the second nine-repeat run alone is 0.0031.
    """
    c = concentration(SPLIT_9B, truths, UND)
    assert c.classes_hit == ["singular_operating_point",
                             "truncated_inner_loop"]
    assert np.isclose(c.p_exact, comb(8, 6) / comb(24, 6))
    assert c.p_corrected < 0.005


def test_the_two_runs_agree_on_classes_but_not_on_cases(truths):
    """The reason the finding is stated at class level.

    If the same cases split every time, the instability would be a property
    of those circuits.  They do not, so it is a property of the kind of
    evidence the class produces.
    """
    a, b = concentration(SPLIT_9, truths, UND), concentration(SPLIT_9B,
                                                              truths, UND)
    assert a.classes_hit == b.classes_hit
    assert set(SPLIT_9) != set(SPLIT_9B)
    assert set(SPLIT_9) & set(SPLIT_9B), "no overlap at all would be odd too"


def test_neither_run_touches_the_undecidable_cases(truths):
    for splits in (SPLIT_9, SPLIT_9B):
        assert concentration(splits, truths, UND).undecidable_overlap == []


def test_correction_is_applied_and_never_exceeds_one(truths):
    c = concentration(SPLIT_9, truths, UND)
    assert c.p_corrected > c.p_exact
    assert c.p_corrected <= 1.0


def test_splits_spread_over_all_classes_are_not_surprising(truths):
    """A control: one case from each class must not come out significant."""
    spread = ["none_0", "kink_0", "neggds_0", "trunc_0", "badstart_0",
              "singular_0"]
    c = concentration(spread, truths, UND)
    assert len(c.classes_hit) == 6
    assert c.p_exact == 1.0 or np.isclose(c.p_exact, 1.0)


def test_undecidable_overlap_is_reported(truths):
    c = concentration(["kink_1", "trunc_0"], truths, UND)
    assert c.undecidable_overlap == ["kink_1"]
    assert concentration(SPLIT_9, truths, UND).undecidable_overlap == []


# --------------------------------------------------------------------------
# Flip fields
# --------------------------------------------------------------------------

def test_flip_fields_never_mutates_the_trace(traces):
    import copy
    before = copy.deepcopy(traces["singular_0"])
    flip_fields(traces["singular_0"])
    assert traces["singular_0"] == before


def test_negative_gds_hangs_on_exactly_one_field(traces, truths):
    """The class that never split is the one a single sign decides."""
    for name, t in traces.items():
        if truths[name] == "negative_gds":
            assert flip_fields(t) == ["gds"], (name, flip_fields(t))


def test_singular_is_the_most_entangled_class(traces, truths):
    """Five conditions have to hold together; it split on three of four."""
    for name, t in traces.items():
        if truths[name] == "singular_operating_point":
            assert len(flip_fields(t)) == 5, (name, flip_fields(t))


def test_flip_fields_are_drawn_from_the_declared_sets(traces):
    allowed = set(TOP_FIELDS) | set(DEVICE_FIELDS)
    for t in traces.values():
        assert set(flip_fields(t)) <= allowed


def test_every_case_has_at_least_one_flipping_field(traces):
    """A diagnosis no single field can move would not be read from evidence."""
    for name, t in traces.items():
        assert flip_fields(t), name


# --------------------------------------------------------------------------
# The confounds are measured, not assumed away
# --------------------------------------------------------------------------

def test_size_confounds_do_not_explain_the_splits(traces, truths):
    """The cheap explanation, ruled out rather than ignored.

    If the model simply wavered on whatever evidence was longest, the two
    classes with full 25-step histories that never split would not exist.
    """
    rows = class_rows(traces, truths, SPLIT_9)
    p = predictors(rows)
    assert abs(p["history_length"]) < 0.3
    assert abs(p["evidence_chars"]) < 0.3
    assert p["flip_count"] > 0.5


def test_classes_that_never_split_have_histories_as_long_as_those_that_do(
        traces, truths):
    rows = {r.name: r for r in class_rows(traces, truths, SPLIT_9)}
    assert rows["bad_initial_guess"].n_split == 0
    assert rows["negative_gds"].n_split == 0
    assert rows["bad_initial_guess"].history_length >= (
        rows["singular_operating_point"].history_length)
    assert rows["negative_gds"].history_length >= (
        rows["truncated_inner_loop"].history_length)


def test_the_known_exception_to_the_flip_count_story(traces, truths):
    """Two classes at three fields, opposite outcomes.

    The README says the correlation has a known exception rather than
    presenting it as a mechanism.  This is that exception, pinned, so the
    caveat cannot quietly stop being true.
    """
    rows = {r.name: r for r in class_rows(traces, truths, SPLIT_9)}
    a, b = rows["truncated_inner_loop"], rows["bad_initial_guess"]
    assert a.flip_count == b.flip_count == 3.0
    assert a.n_split > 0 and b.n_split == 0


def test_class_rows_cover_every_class(traces, truths):
    rows = class_rows(traces, truths, SPLIT_9)
    assert {r.name for r in rows} == set(truths.values())
    assert all(r.n == 4 for r in rows)


def test_rows_are_ordered_by_entanglement(traces, truths):
    counts = [r.flip_count for r in class_rows(traces, truths, SPLIT_9)]
    assert counts == sorted(counts)


# --------------------------------------------------------------------------
# Spearman
# --------------------------------------------------------------------------

def test_spearman_is_one_for_a_monotonic_pair():
    assert np.isclose(_spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert np.isclose(_spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)


def test_spearman_is_nan_without_variation():
    assert np.isnan(_spearman([1, 1, 1, 1], [1, 2, 3, 4]))
    assert np.isnan(_spearman([1, 2], [2, 1]))


# --------------------------------------------------------------------------
# The summary the report reads
# --------------------------------------------------------------------------

def test_summarise_has_everything_the_report_quotes(traces, truths):
    s = summarise(traces, truths, SPLIT_9, UND)
    assert set(s) == {"classes", "predictors", "concentration", "flip_fields"}
    co = s["concentration"]
    for key in ("split_cases", "classes_hit", "cases_in_those_classes",
                "p_exact", "p_corrected", "n_class_groups",
                "undecidable_overlap", "p_miss_undecidable"):
        assert key in co
    assert len(s["classes"]) == 6
    assert set(s["flip_fields"]) == set(traces)


def test_summarise_is_json_serialisable(traces, truths):
    import json
    json.dumps(summarise(traces, truths, SPLIT_9, UND))
