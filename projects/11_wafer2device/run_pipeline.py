#!/usr/bin/env python3
"""Run the wafer2device pipeline.

Examples
--------
    python run_pipeline.py                        # rule path, strict gate
    python run_pipeline.py --mode stub            # exercise the adjudication layer
    python run_pipeline.py --approval interactive # review each case by hand
    python run_pipeline.py --cases edge_ring_plus_center_00
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wafer2device.gate import load_yaml
from wafer2device.orchestrator import Pipeline
from wafer2device.s1_pattern.data import load_cases
from wafer2device.s5_report.report import write_case_report, write_summary

ROOT = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/pipeline.yaml"))
    ap.add_argument("--checks", default=str(ROOT / "configs/checks.yaml"))
    ap.add_argument("--whitelist", default=str(ROOT / "configs/whitelist.yaml"))
    ap.add_argument("--mode", choices=["rule", "llm"],
                    help="hypothesis mode override")
    ap.add_argument("--provider", choices=["cache", "openai", "stub"],
                    help="LLM provider override")
    ap.add_argument("--approval", choices=["auto", "strict", "interactive"],
                    help="approval gate mode override")
    ap.add_argument("--cases", nargs="*", help="only run these case ids")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    checks_cfg = load_yaml(args.checks)
    wl_cfg = load_yaml(args.whitelist)

    if args.mode:
        cfg["hypothesis"]["mode"] = args.mode
    if args.provider:
        cfg["hypothesis"]["mode"] = "llm"
        cfg["hypothesis"]["llm"]["provider"] = args.provider
    if args.approval:
        cfg["approval"]["mode"] = args.approval

    cases = load_cases(cfg)
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in cases if c.case_id in wanted]
        if not cases:
            print(f"no cases matched {sorted(wanted)}", file=sys.stderr)
            return 2

    pipe = Pipeline(cfg, checks_cfg, wl_cfg)
    if pipe.secondary is None and cfg["device_model"].get("secondary"):
        print(f"[warn] secondary model unavailable: {pipe._secondary_note}")

    done = []
    for case in cases:
        pipe.run_case(case)
        write_case_report(case, cfg["report"]["output_dir"])
        done.append(case)
        if not args.quiet:
            gate = case.approval.decision if case.approval else "-"
            src = case.hypothesis.value.source if case.hypothesis else "-"
            comp = (f"{case.compensation.value.improvement:+.1%}"
                    if case.compensation else "skipped")
            print(f"{case.case_id:<28} gate={gate:<10} source={src:<16} "
                  f"rounds={len(case.replan_log)}  compensation={comp}")

    summary = write_summary(done, cfg["report"]["output_dir"])
    print(f"\nreports -> {cfg['report']['output_dir']}/  (summary: {summary})")
    print(f"replan logs -> {cfg['report']['log_dir']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
