"""The physics approval gate.

The pipeline stops here.  A hypothesis is an *assumption about physics*, and in
this project a machine is not allowed to promote an assumption into an input for
the device model on its own.

Three modes:

``strict``      auto-approve only when every physics check passed; otherwise the
                case is escalated and marked as such.  This is the default.
``interactive`` present the assumption card and ask the engineer.
``auto``        approve unconditionally -- batch experiments only, and the
                resulting report says so.

Whatever happens, an ``ApprovalRecord`` is written, so a reader of the output can
always tell whether a human looked at a given case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .contracts import ApprovalRecord, CheckReport, Hypothesis, WaferCase


def assumption_card(case: WaferCase, h: Hypothesis,
                    report: CheckReport) -> Dict[str, Any]:
    """Human-readable record of what is about to be assumed, and on what basis."""
    card: Dict[str, Any] = {
        "case_id": case.case_id,
        "wafer_map_provenance": case.wafer_map.provenance.value,
        "pattern": case.pattern.value.to_dict() if case.pattern else None,
        "descriptors": case.descriptors.value.to_dict() if case.descriptors else None,
        "rule_hypothesis": (case.rule_hypothesis.value.to_dict()
                            if case.rule_hypothesis else None),
        "proposed_hypothesis": h.to_dict(),
        "physics_checks": report.to_dict(),
        "replan_rounds": len(case.replan_log),
    }
    return card


def write_card(card: Dict[str, Any], out_dir: str, case_id: str) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"assumption_card_{case_id}.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(card, fh, allow_unicode=True, sort_keys=False)
    return path


def render_card(card: Dict[str, Any]) -> str:
    lines = [f"=== assumption card: {card['case_id']} ===",
             f"wafer map provenance : {card['wafer_map_provenance']}"]
    pat = card.get("pattern") or {}
    lines.append(f"pattern              : {pat.get('label')} "
                 f"(composite={pat.get('is_composite')}, "
                 f"borderline={pat.get('is_borderline')})")
    h = card["proposed_hypothesis"]
    lines.append(f"source               : {h['source']} "
                 f"(adjudication={h.get('adjudication')}, "
                 f"replan rounds={card['replan_rounds']})")
    lines.append("proposed variation terms:")
    for t in h["terms"]:
        lines.append(f"  - {t['param']:<26} {t['spatial_form']:<18} "
                     f"{t['magnitude']:+.4f}  (conf {t['confidence']:.2f})")
    lines.append(f"rationale            : {h['rationale']}")
    lines.append("physics checks:")
    for r in card["physics_checks"]["results"]:
        flag = "PASS" if r["passed"] else "FAIL"
        lines.append(f"  [{flag}] {r['name']:<22} {r['detail']}")
        if not r["passed"] and r["direction_hint"]:
            lines.append(f"         hint: {r['direction_hint']}")
    return "\n".join(lines)


def request_approval(case: WaferCase, h: Hypothesis, report: CheckReport,
                     mode: str, out_dir: str) -> ApprovalRecord:
    card = assumption_card(case, h, report)
    write_card(card, out_dir, case.case_id)

    if mode == "auto":
        return ApprovalRecord(decision="approved", approver="auto",
                              comment="approval.mode=auto (batch run; no human review)")

    if mode == "strict":
        if report.passed:
            return ApprovalRecord(decision="approved", approver="auto-strict",
                                  comment="all physics checks passed")
        return ApprovalRecord(
            decision="escalated", approver="auto-strict",
            comment="physics checks failed after replanning; human review required")

    # interactive
    print(render_card(card))
    print("\n[a]pprove  [m]odify  [r]eject  [s]kip(escalate)")
    choice = input("decision> ").strip().lower()[:1]
    if choice == "a":
        return ApprovalRecord(decision="approved", comment=input("comment> ").strip())
    if choice == "m":
        print("Enter magnitudes, comma separated, in the order shown above:")
        raw = input("magnitudes> ").strip()
        try:
            mags = [float(x) for x in raw.split(",")]
            for t, m in zip(h.terms, mags):
                t.magnitude = m
            return ApprovalRecord(decision="modified",
                                  comment="magnitudes overridden by engineer",
                                  modified_terms=[t.to_dict() for t in h.terms])
        except ValueError:
            print("could not parse; escalating instead")
            return ApprovalRecord(decision="escalated", comment="bad manual input")
    if choice == "r":
        return ApprovalRecord(decision="rejected", comment=input("reason> ").strip())
    return ApprovalRecord(decision="escalated", comment="skipped by engineer")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
