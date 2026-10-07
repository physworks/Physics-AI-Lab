"""Where the LLM arm's instability actually falls, and what predicts it.

The nine-repeat run split its vote on 5 of 24 cases and every one of them was
in two of the six classes.  That is not a shape sampling noise produces, so
this module measures the properties that might explain it instead of asserting
one.

Three quantities, all computed from the same traces the diagnosers see:

``flip_fields``
    For each case, which single-field changes to the evidence alter the rule's
    answer.  A class whose diagnosis hinges on one field is a different kind
    of problem from one that needs a conjunction of five.

``concentration``
    The chance that splits placed uniformly at random would land entirely
    inside the classes they did, with a correction for the fact that any pair
    of classes would have looked equally surprising after the fact.

``confounds``
    Evidence length and payload size per class, because "the model wavers on
    the messy ones" is the cheaper explanation and has to be ruled out before
    any claim about logical structure is worth making.

Nothing here decides anything.  It reports numbers that the report generator
and the README quote, so that neither has to take a reading of the rule's
source code on faith.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from math import comb
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .diagnose import evidence_for, rule_diagnose

#: Top-level trace booleans a one-field perturbation can flip.
TOP_FIELDS: Tuple[str, ...] = ("converged", "retry_from_default_converged",
                               "started_far_from_default", "oscillating",
                               "stalled", "diverging")

#: Device-state fields, perturbed in the direction that changes their meaning.
DEVICE_FIELDS: Tuple[str, ...] = ("gds", "gm", "inner_converged")


def _perturb(trace: Dict, field_name: str, where: str) -> Dict:
    """One field changed, everything else held.  Never mutates the input."""
    t = copy.deepcopy(trace)
    if where == "top":
        v = t.get(field_name)
        if isinstance(v, bool):
            t[field_name] = not v
        return t

    states = t.get("device_state") or []
    if not states:
        return t
    d = states[0]
    v = d.get(field_name)
    if field_name == "gds":
        v = float(v or 0.0)
        d[field_name] = (-abs(v) - 1e-3) if v >= 0.0 else abs(v)
    elif field_name == "gm":
        d[field_name] = 0.0 if float(v or 0.0) != 0.0 else 1.0
    else:
        d[field_name] = not bool(v)
    return t


def flip_fields(trace: Dict) -> List[str]:
    """Fields whose single-handed change moves the rule off its answer."""
    base = rule_diagnose(trace)
    out = []
    for f in TOP_FIELDS:
        if rule_diagnose(_perturb(trace, f, "top")) != base:
            out.append(f)
    for f in DEVICE_FIELDS:
        if rule_diagnose(_perturb(trace, f, "dev")) != base:
            out.append(f)
    return out


@dataclass
class ClassRow:
    name: str
    n: int
    n_split: int
    flip_count: float
    evidence_chars: float
    history_length: float

    @property
    def split_rate(self) -> float:
        return self.n_split / self.n if self.n else float("nan")


@dataclass
class Concentration:
    """How surprising the observed placement of splits is."""

    split_cases: List[str]
    classes_hit: List[str]
    cases_in_those_classes: int
    n_cases: int
    p_exact: float
    p_corrected: float
    n_class_pairs: int
    undecidable_overlap: List[str] = field(default_factory=list)
    p_miss_undecidable: float = float("nan")


def _p_all_inside(k: int, subset: int, n: int) -> float:
    if k == 0 or k > subset:
        return float("nan") if k == 0 else 0.0
    return comb(subset, k) / comb(n, k)


def _p_none_inside(k: int, subset: int, n: int) -> float:
    if k == 0 or k > n - subset:
        return float("nan") if k == 0 else 0.0
    return comb(n - subset, k) / comb(n, k)


def concentration(split_cases: Sequence[str], truths: Dict[str, str],
                  undecidable: Iterable[str] = ()) -> Concentration:
    """Probability the splits would land where they did, if placed at random.

    The exact figure answers "all of them inside *these* classes".  Since any
    equally small group of classes would have looked just as striking once
    seen, it is multiplied by the number of class groups of that size -- a
    Bonferroni correction, and the number worth quoting.
    """
    splits = list(split_cases)
    n = len(truths)
    hit = sorted({truths[c] for c in splits if c in truths})
    size = sum(1 for t in truths.values() if t in hit)
    k = len(splits)

    p = _p_all_inside(k, size, n)
    all_classes = sorted(set(truths.values()))
    pairs = comb(len(all_classes), len(hit)) if hit else 0
    und = sorted(set(splits) & set(undecidable))

    return Concentration(
        split_cases=splits, classes_hit=hit, cases_in_those_classes=size,
        n_cases=n, p_exact=p,
        p_corrected=min(1.0, p * pairs) if pairs and p == p else float("nan"),
        n_class_pairs=pairs, undecidable_overlap=und,
        p_miss_undecidable=_p_none_inside(k, len(list(undecidable)), n))


def class_rows(traces: Dict[str, Dict], truths: Dict[str, str],
               split_cases: Sequence[str] = ()) -> List[ClassRow]:
    """Per-class flip count, split rate, and the two size confounds."""
    splits = set(split_cases)
    buckets: Dict[str, List[str]] = {}
    for name in traces:
        buckets.setdefault(truths[name], []).append(name)

    rows = []
    for cls, names in buckets.items():
        flips, chars, hist = [], [], []
        for nm in names:
            t = traces[nm]
            flips.append(len(flip_fields(t)))
            ev = evidence_for(t)
            chars.append(len(str(ev)))
            hist.append(len(ev.get("residual_history") or []))
        rows.append(ClassRow(
            name=cls, n=len(names),
            n_split=sum(1 for nm in names if nm in splits),
            flip_count=float(np.mean(flips)),
            evidence_chars=float(np.mean(chars)),
            history_length=float(np.mean(hist))))
    return sorted(rows, key=lambda r: r.flip_count)


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def predictors(rows: Sequence[ClassRow]) -> Dict[str, float]:
    """Rank correlation of each candidate explanation with the split rate.

    Reported together on purpose.  A high correlation for one of them is only
    worth reading next to a near-zero one for the cheap alternatives.
    """
    rate = [r.split_rate for r in rows]
    return {
        "flip_count": _spearman([r.flip_count for r in rows], rate),
        "history_length": _spearman([r.history_length for r in rows], rate),
        "evidence_chars": _spearman([r.evidence_chars for r in rows], rate),
    }


def summarise(traces: Dict[str, Dict], truths: Dict[str, str],
              split_cases: Sequence[str],
              undecidable: Iterable[str] = ()) -> Dict:
    rows = class_rows(traces, truths, split_cases)
    conc = concentration(split_cases, truths, undecidable)
    return {
        "classes": [{"class": r.name, "n": r.n, "n_split": r.n_split,
                     "split_rate": r.split_rate,
                     "mean_flip_fields": r.flip_count,
                     "mean_evidence_chars": r.evidence_chars,
                     "mean_history_length": r.history_length} for r in rows],
        "predictors": predictors(rows),
        "concentration": {
            "split_cases": conc.split_cases,
            "classes_hit": conc.classes_hit,
            "cases_in_those_classes": conc.cases_in_those_classes,
            "p_exact": conc.p_exact,
            "p_corrected": conc.p_corrected,
            "n_class_groups": conc.n_class_pairs,
            "undecidable_overlap": conc.undecidable_overlap,
            "p_miss_undecidable": conc.p_miss_undecidable,
        },
        "flip_fields": {nm: flip_fields(t) for nm, t in traces.items()},
    }
