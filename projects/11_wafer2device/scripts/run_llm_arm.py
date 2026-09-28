#!/usr/bin/env python3
"""Run the live-LLM arm and print everything needed to update the write-ups.

Run this on a machine that can reach api.openai.com:

    export OPENAI_API_KEY=sk-...        # Windows: set OPENAI_API_KEY=sk-...
    python scripts/run_llm_arm.py

It does four things:

1. checks the key and the connection before spending anything
2. runs the comparison with the live API
3. refuses to report a comparison if the calls silently fell back to the rules
4. prints the exact numbers to paste into the README and the portfolio

The LLM responses are written to ``llm_cache/``.  Commit that directory and a
reviewer can reproduce the same table with ``--arm cache`` and no API key.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def preflight(model: str) -> bool:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY is not set.")
        print('  macOS/Linux:  export OPENAI_API_KEY=sk-...')
        print('  Windows CMD:  set OPENAI_API_KEY=sk-...')
        return False
    print(f"key      : ...{key[-6:]} (length {len(key)})")
    try:
        from openai import OpenAI
    except ImportError:
        print("The openai package is missing.  pip install openai")
        return False
    try:
        client = OpenAI(api_key=key)
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "Reply with JSON only."},
                      {"role": "user", "content": '{"ping": true}'}],
            response_format={"type": "json_object"}, temperature=0)
        print(f"connection: ok (served by {r.model})")
        return True
    except Exception as exc:                            # noqa: BLE001
        print(f"connection: FAILED -- {type(exc).__name__}: {str(exc)[:200]}")
        return False


def pct(x: float) -> str:
    return f"{x:.0%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--n-per-scenario", type=int, default=3)
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    print("=" * 66)
    print("wafer2device -- live LLM arm")
    print("=" * 66)

    if not args.skip_preflight and not preflight(args.model):
        print("\nAborted before spending anything.")
        return 1

    # Point the pipeline at the live API for this run only; the file on disk is
    # left untouched so the committed default stays offline-safe.
    cfg_path = ROOT / "configs/pipeline.yaml"
    original = cfg_path.read_text(encoding="utf-8")
    patched = (original
               .replace("allow_network: false", "allow_network: true")
               .replace("model: gpt-4o-mini", f"model: {args.model}"))
    cfg_path.write_text(patched, encoding="utf-8")

    try:
        print("\nrunning comparison (this makes real API calls)...\n")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "experiments/compare_rule_vs_llm.py"),
             "--arm", "openai", "--n-per-scenario", str(args.n_per_scenario)],
            cwd=str(ROOT))
    finally:
        cfg_path.write_text(original, encoding="utf-8")
        print("\nconfigs/pipeline.yaml restored to its committed state.")

    if proc.returncode != 0:
        print("\nThe run did not produce a valid comparison. Nothing to report.")
        print("Keep using the offline numbers and say so in the write-up.")
        return proc.returncode

    raw = json.loads((ROOT / "outputs/experiment_rule_vs_llm.json").read_text())
    prov = raw["provenance"]

    import numpy as np

    def agg(rows, kind, key):
        sel = [r for r in rows if kind == "all" or r["kind"] == kind]
        vals = [r[key] for r in sel if r[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n" + "=" * 66)
    print("NUMBERS TO PASTE")
    print("=" * 66)
    print(f"model: {args.model}   |   provenance valid: {prov['valid']}")
    print(f"hypothesis sources: {prov['counts']}\n")

    for kind in ("simple", "composite", "borderline"):
        r, t = raw["rule"], raw["treatment"]
        print(f"[{kind}]")
        print(f"  param F1        rule {agg(r,kind,'param_f1'):.3f}  ->  "
              f"llm {agg(t,kind,'param_f1'):.3f}")
        print(f"  escalation      rule {pct(agg(r,kind,'escalated'))}  ->  "
              f"llm {pct(agg(t,kind,'escalated'))}")
        print(f"  out-of-spec red rule {agg(r,kind,'compensation'):.1%}  ->  "
              f"llm {agg(t,kind,'compensation'):.1%}")
        print(f"  magnitude MAE   rule {agg(r,kind,'magnitude_mae'):.4f}  ->  "
              f"llm {agg(t,kind,'magnitude_mae'):.4f}")
        print()

    cache = list((ROOT / "llm_cache").glob("*.json"))
    print(f"cached responses: {len(cache)} files in llm_cache/")
    print("\nNext:")
    print("  1. git add llm_cache && git commit -m 'add LLM response cache'")
    print("  2. update the README table and the portfolio with the numbers above")
    print("  3. rotate the API key if it was ever pasted somewhere shared")
    print("\nIf the LLM did NOT beat the rules, report that result as it is.")
    print("'We tried it, the rules won here, so the rules stayed the default'")
    print("is a stronger claim than a table where the AI always wins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
