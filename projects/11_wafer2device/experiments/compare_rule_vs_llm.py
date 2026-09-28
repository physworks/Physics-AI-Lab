#!/usr/bin/env python3
"""Does the adjudication layer actually beat the rules?

This is the experiment that answers the obvious objection -- "a rule table can
already map a wafer pattern to a process cause, so why spend an LLM call on it?"
The rule engine is the control arm and it is deliberately a strong one.

Metrics, scored per case kind (simple / composite / borderline):

    first_pass        physics checks passed on round 0 (no replanning needed)
    rounds            mean number of replanning rounds consumed
    escalated         fraction handed back to the engineer
    param_f1          F1 of identified (param, spatial_form) pairs vs ground truth
    magnitude_mae     mean absolute magnitude error on correctly identified terms
    compensation      mean out-of-spec reduction achieved downstream

Only synthetic cases are scored, because only they have a ground truth.  Real
WM-811K maps have no process ground truth at all -- which is the whole reason
the hypothesis is an assumption behind a human gate.

Usage
-----
    python experiments/compare_rule_vs_llm.py                 # rule vs stub
    python experiments/compare_rule_vs_llm.py --arm openai    # rule vs real LLM
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from wafer2device.gate import load_yaml
from wafer2device.orchestrator import Pipeline
from wafer2device.s1_pattern.data import generate_dataset


def score_case(case) -> Dict[str, Any]:
    gt = case.ground_truth
    truth = {(t["param"], t["spatial_form"]) for t in gt["terms"]}
    truth_mag = {(t["param"], t["spatial_form"]): t["magnitude"] for t in gt["terms"]}

    h = case.hypothesis.value if case.hypothesis else None
    pred = {(t.param, t.spatial_form) for t in h.terms} if h else set()
    pred_mag = {(t.param, t.spatial_form): t.magnitude for t in h.terms} if h else {}

    tp = len(truth & pred)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(truth) if truth else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    errs = [abs(pred_mag[k] - truth_mag[k]) for k in (truth & pred)]
    first = case.replan_log[0].check_report.passed if case.replan_log else False

    # Record where every hypothesis in the loop came from.  Without this, an arm
    # whose API calls all failed would fall back to the rule engine and report
    # itself as an LLM result -- the two arms would be identical and the table
    # would say so in a way nobody could detect after the fact.
    sources = [a.hypothesis.source for a in case.replan_log]

    return {
        "case_id": case.case_id,
        "kind": gt["kind"],
        "first_pass": bool(first),
        "rounds": len(case.replan_log),
        "escalated": bool(case.escalated),
        "param_f1": f1,
        "magnitude_mae": float(np.mean(errs)) if errs else None,
        "compensation": (case.compensation.value.improvement
                         if case.compensation else None),
        "final_source": h.source if h else None,
        "sources": sources,
    }


def provenance_check(rows: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    """Verify the treatment arm actually ran the thing it claims to have run."""
    all_sources = [s for r in rows for s in r["sources"]]
    n = len(all_sources)
    counts: Dict[str, int] = {}
    for s in all_sources:
        counts[s] = counts.get(s, 0) + 1

    llm_like = sum(v for k, v in counts.items() if k.startswith("llm"))
    stub_like = counts.get("stub_adjudicator", 0)
    fallback = counts.get("rule_fallback", 0)
    plain_rule = counts.get("rule", 0)

    if arm in ("openai", "cache"):
        expected, got = "llm*", llm_like
    elif arm == "stub":
        expected, got = "stub_adjudicator", stub_like
    else:
        expected, got = "rule", plain_rule

    return {"arm": arm, "n_hypotheses": n, "counts": counts,
            "expected_source": expected, "n_expected": got,
            "n_rule_fallback": fallback,
            "valid": got > 0 and fallback < n}


def run_arm(name: str, cfg: Dict[str, Any], checks_cfg: Dict[str, Any],
            wl_cfg: Dict[str, Any], n_per: int) -> List[Dict[str, Any]]:
    pipe = Pipeline(cfg, checks_cfg, wl_cfg)
    rows = []
    for case in generate_dataset(n_per_scenario=n_per,
                                 n=int(cfg["data"]["grid"]),
                                 seed=int(cfg["data"]["seed"])):
        pipe.run_case(case)
        rows.append(score_case(case))
    return rows


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    kinds = sorted({r["kind"] for r in rows}) + ["all"]
    for kind in kinds:
        sel = rows if kind == "all" else [r for r in rows if r["kind"] == kind]
        maes = [r["magnitude_mae"] for r in sel if r["magnitude_mae"] is not None]
        comps = [r["compensation"] for r in sel if r["compensation"] is not None]
        out[kind] = {
            "n": len(sel),
            "first_pass": float(np.mean([r["first_pass"] for r in sel])),
            "rounds": float(np.mean([r["rounds"] for r in sel])),
            "escalated": float(np.mean([r["escalated"] for r in sel])),
            "param_f1": float(np.mean([r["param_f1"] for r in sel])),
            "magnitude_mae": float(np.mean(maes)) if maes else float("nan"),
            "compensation": float(np.mean(comps)) if comps else float("nan"),
        }
    return out


def render(agg_rule: Dict[str, Any], agg_llm: Dict[str, Any],
           arm_label: str) -> str:
    L = ["# Rule vs adjudicated hypothesis", "",
         f"Control arm: **deterministic rule engine**. "
         f"Treatment arm: **{arm_label}**.", "",
         "| case kind | n | metric | rule | " + arm_label + " |",
         "|---|---|---|---|---|"]
    metrics = [("first_pass", "first-pass check rate", "{:.0%}"),
               ("rounds", "mean replan rounds", "{:.2f}"),
               ("escalated", "escalation rate", "{:.0%}"),
               ("param_f1", "param F1 vs ground truth", "{:.3f}"),
               ("magnitude_mae", "magnitude MAE", "{:.4f}"),
               ("compensation", "mean out-of-spec reduction", "{:.1%}")]
    for kind in agg_rule:
        for key, label, fmt in metrics:
            a, b = agg_rule[kind][key], agg_llm[kind][key]
            fa = "n/a" if isinstance(a, float) and np.isnan(a) else fmt.format(a)
            fb = "n/a" if isinstance(b, float) and np.isnan(b) else fmt.format(b)
            L.append(f"| {kind} | {agg_rule[kind]['n']} | {label} | {fa} | {fb} |")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["stub", "cache", "openai"], default="stub")
    ap.add_argument("--n-per-scenario", type=int, default=3)
    ap.add_argument("--out", default="outputs/experiment_rule_vs_llm.md")
    args = ap.parse_args()

    cfg = load_yaml(str(ROOT / "configs/pipeline.yaml"))
    checks_cfg = load_yaml(str(ROOT / "configs/checks.yaml"))
    wl_cfg = load_yaml(str(ROOT / "configs/whitelist.yaml"))
    cfg["approval"]["mode"] = "strict"        # identical gate policy for both arms

    rule_cfg = json.loads(json.dumps(cfg))
    rule_cfg["hypothesis"]["mode"] = "rule"
    rows_rule = run_arm("rule", rule_cfg, checks_cfg, wl_cfg, args.n_per_scenario)

    llm_cfg = json.loads(json.dumps(cfg))
    llm_cfg["hypothesis"]["mode"] = "llm"
    llm_cfg["hypothesis"]["llm"]["provider"] = args.arm
    if args.arm == "openai":
        llm_cfg["hypothesis"]["llm"]["allow_network"] = True
    rows_llm = run_arm(args.arm, llm_cfg, checks_cfg, wl_cfg, args.n_per_scenario)

    # The offline stub is NOT a language model and is never reported as one.
    arm_label = {"stub": "reference adjudicator (NOT an LLM)",
                 "cache": "LLM (cached responses)",
                 "openai": "LLM (live API)"}[args.arm]

    prov = provenance_check(rows_llm, args.arm)
    text = render(aggregate(rows_rule), aggregate(rows_llm), arm_label)
    text += ("\n## Provenance check\n\n"
             f"- treatment arm: `{args.arm}` (expected hypothesis source "
             f"`{prov['expected_source']}`)\n"
             f"- hypotheses generated: {prov['n_hypotheses']}, "
             f"from the expected source: {prov['n_expected']}, "
             f"rule fallback: {prov['n_rule_fallback']}\n"
             f"- source counts: `{prov['counts']}`\n"
             f"- **valid: {prov['valid']}**\n")
    if not prov["valid"]:
        text += ("\n> The treatment arm did not actually run. Every hypothesis fell "
                 "back to the rule engine, so the two columns above describe the "
                 "same system and must not be reported as a comparison.\n")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    raw = out.with_suffix(".json")
    raw.write_text(json.dumps({"arm": args.arm, "arm_label": arm_label,
                               "provenance": prov,
                               "rule": rows_rule, "treatment": rows_llm},
                              indent=2), encoding="utf-8")
    print(text)
    print(f"written -> {out}")
    if not prov["valid"]:
        print("\nERROR: treatment arm never ran; see the provenance check above.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
