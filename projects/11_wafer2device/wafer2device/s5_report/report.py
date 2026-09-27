"""S5 -- engineer-facing report.

The report is organised by *provenance*, not by pipeline stage.  A reader sees
immediately which lines are measurements, which are assumptions that a human
approved, and which are model output.  A case that was escalated says so at the
top instead of quietly presenting an optimisation result nobody signed off on.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..contracts import Provenance, WaferCase


def render_case(case: WaferCase) -> str:
    L: List[str] = []
    L.append(f"# Case report: {case.case_id}")
    L.append("")

    if case.escalated:
        L.append("> **ESCALATED -- no approved hypothesis.** The physics checks were "
                 "not satisfied within the replanning budget, so the device and "
                 "compensation stages were not run. This is a designed outcome: an "
                 "unexplained wafer is handed back to the engineer rather than "
                 "pushed through with a hypothesis nobody trusts.")
        L.append("")

    # -- measured ---------------------------------------------------------- #
    prov = case.wafer_map.provenance
    L.append(f"## 1. Observed ({prov.value})")
    if prov is Provenance.SYNTHETIC:
        L.append("_Synthetic wafer map with a known ground truth; used for scoring._")
    elif prov is Provenance.MEASURED:
        L.append("_Real wafer map. No process ground truth exists for these maps._")
    L.append("")
    if case.descriptors:
        d = case.descriptors.value
        L.append("| descriptor | value |")
        L.append("|---|---|")
        L.append(f"| defect rate | {d.defect_rate:.4f} |")
        L.append(f"| edge/center ratio | {d.edge_center_ratio:.3f} |")
        L.append(f"| radial slope | {d.radial_slope:+.4f} |")
        L.append(f"| angular anisotropy | {d.angular_anisotropy:.3f} |")
        L.append(f"| cluster centroid r | {d.cluster_centroid_r:.3f} |")
        L.append(f"| cluster compactness | {d.cluster_compactness:.3f} |")
        L.append(f"| dies | {d.n_dies} |")
        L.append("")
    if case.pattern:
        p = case.pattern.value
        L.append(f"Rule classification: **{p.label}** "
                 f"(composite={p.is_composite}, borderline={p.is_borderline})")
        L.append("")
        L.append("| pattern | score |")
        L.append("|---|---|")
        for k, v in sorted(p.scores.items(), key=lambda kv: -kv[1]):
            L.append(f"| {k} | {v:.3f} |")
        L.append("")
        for note in p.margin_notes:
            L.append(f"- margin note: {note}")
        if p.margin_notes:
            L.append("")

    # -- assumed ----------------------------------------------------------- #
    L.append("## 2. Assumed (hypothesis -- NOT measured)")
    L.append("")
    L.append("_The mapping from a spatial defect signature to a process variation is "
             "a physical hypothesis. It is generated, validated against physics "
             "checks, and approved at the gate before any downstream stage uses it._")
    L.append("")
    if case.rule_hypothesis:
        rh = case.rule_hypothesis.value
        L.append(f"**Rule baseline** ({rh.source}):")
        for t in rh.terms:
            L.append(f"- `{t.param}` / `{t.spatial_form}` = {t.magnitude:+.4f} "
                     f"(conf {t.confidence:.2f})")
        L.append(f"- rationale: {rh.rationale}")
        L.append("")
    if case.hypothesis:
        h = case.hypothesis.value
        L.append(f"**Final hypothesis** (source=`{h.source}`, "
                 f"adjudication=`{h.adjudication}`, replan round {h.replan_round}):")
        for t in h.terms:
            L.append(f"- `{t.param}` / `{t.spatial_form}` = {t.magnitude:+.4f} "
                     f"(conf {t.confidence:.2f})")
        L.append(f"- rationale: {h.rationale}")
        L.append("")
    if case.approval:
        a = case.approval
        L.append(f"**Gate decision:** `{a.decision}` by {a.approver} -- {a.comment}")
        L.append("")

    # -- replanning audit trail ------------------------------------------- #
    if case.replan_log:
        L.append("### Replanning loop")
        L.append("")
        L.append("| round | source | violation score | passed | stop reason |")
        L.append("|---|---|---|---|---|")
        for a in case.replan_log:
            L.append(f"| {a.round_index} | {a.hypothesis.source} | "
                     f"{a.violation_score:.3f} | {a.check_report.passed} | "
                     f"{a.stop_reason or ''} |")
        L.append("")

    if case.check_report:
        L.append("### Physics checks (final)")
        L.append("")
        L.append("| check | result | observed | threshold | detail |")
        L.append("|---|---|---|---|---|")
        for r in case.check_report.results:
            L.append(f"| {r.name} | {'PASS' if r.passed else 'FAIL'} | "
                     f"{r.observed:.4f} | {r.threshold:.4f} | {r.detail} |")
        L.append("")
        for r in case.check_report.failures():
            if r.direction_hint:
                L.append(f"- direction hint ({r.name}): {r.direction_hint}")
        L.append("")

    # -- predicted / optimized -------------------------------------------- #
    if case.device_response:
        r = case.device_response.value
        L.append("## 3. Predicted (device model output)")
        L.append("")
        L.append(f"Model: `{r.model_name}` -- out-of-spec rate "
                 f"**{r.out_of_spec_rate:.3f}**")
        L.append("")
        L.append("| metric | value |")
        L.append("|---|---|")
        for k, v in r.stats.items():
            L.append(f"| {k} | {v:.6g} |")
        L.append("")

    if case.compensation:
        c = case.compensation.value
        L.append("## 4. Optimized (compensation)")
        L.append("")
        L.append(f"Out-of-spec rate {c.baseline_out_of_spec:.3f} -> "
                 f"{c.optimized_out_of_spec:.3f} (**{c.improvement:+.1%}**) "
                 f"over {c.n_evaluations} evaluations.")
        L.append("")
        L.append("| knob | value |")
        L.append("|---|---|")
        for k, v in c.knobs.items():
            L.append(f"| {k} | {v:+.5f} |")
        L.append("")

    if case.ground_truth:
        L.append("## 5. Ground truth (synthetic cases only)")
        L.append("")
        L.append(f"Scenario `{case.ground_truth['scenario']}` "
                 f"(kind: {case.ground_truth['kind']})")
        for t in case.ground_truth["terms"]:
            L.append(f"- `{t['param']}` / `{t['spatial_form']}` = "
                     f"{t['magnitude']:+.4f}")
        L.append("")

    L.append("## Log")
    L.append("")
    for note in case.notes:
        L.append(f"- {note}")
    L.append("")
    return "\n".join(L)


def write_case_report(case: WaferCase, out_dir: str) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"report_{case.case_id}.md"
    path.write_text(render_case(case), encoding="utf-8")
    return path


def write_summary(cases: List[WaferCase], out_dir: str) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    L = ["# Run summary", "",
         "| case | provenance | pattern | source | rounds | gate | out-of-spec (base -> opt) |",
         "|---|---|---|---|---|---|---|"]
    for c in cases:
        pat = c.pattern.value.label if c.pattern else "-"
        src = c.hypothesis.value.source if c.hypothesis else "-"
        gate = c.approval.decision if c.approval else "-"
        if c.compensation:
            comp = (f"{c.compensation.value.baseline_out_of_spec:.3f} -> "
                    f"{c.compensation.value.optimized_out_of_spec:.3f}")
        else:
            comp = "not run"
        L.append(f"| {c.case_id} | {c.wafer_map.provenance.value} | {pat} | {src} | "
                 f"{len(c.replan_log)} | {gate} | {comp} |")
    L.append("")
    n_esc = sum(1 for c in cases if c.escalated)
    L.append(f"Escalated to engineer: **{n_esc}/{len(cases)}** cases.")
    L.append("")
    path = d / "summary.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
