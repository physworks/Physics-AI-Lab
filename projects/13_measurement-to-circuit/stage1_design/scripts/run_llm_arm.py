#!/usr/bin/env python3
"""Fill in the agent row of the decision-boundary table, on your own machine.

The sandbox this project was developed in cannot reach ``api.openai.com``, so
the committed ``outputs/`` has a rule fallback in the agent row and says so.
This script is how that row gets filled with an actual model.

    export OPENAI_API_KEY=...     # or ANTHROPIC_API_KEY, or GEMINI_API_KEY
    python scripts/run_llm_arm.py --model gpt-4o-mini          --repeats 3
    python scripts/run_llm_arm.py --model claude-sonnet-4-5    --repeats 3
    python scripts/run_llm_arm.py --model gemini-3.5-flash-lite --repeats 3

A key may also live in a ``.env`` file beside the project; it is gitignored,
because this repository is public.  Model names age faster than this script,
so ``--list-models`` asks the provider what the key can actually reach.

It asks the model, at every trust level, the one question the agent arm owns:
estimate ``theta`` or fix it.  The ground-truth boundary is already measured
in ``outputs/results.json``, so the answer is scored, not admired.

``--repeats`` runs the whole sweep more than once.  A decision-maker that
gives a different boundary on each run does not have a boundary, and the
report should say so rather than quoting whichever run looked best.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oed import DesignConstraints, DeviceParams, NoiseModel, build_pool, decide
from oed.agent import (api_key_for, key_env_names, list_models, load_dotenv,
                       provider_for)
from oed.pipeline import screen
from oed.scenarios import SUBJECT, breakeven_sweep
from run_design import constraints


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="gpt-4o-mini, claude-sonnet-4-5, ...")
    ap.add_argument("--provider", default=None,
                    choices=["openai", "anthropic", "gemini"],
                    help="inferred from --model when omitted")
    ap.add_argument("--prompt-version", type=int, default=2, choices=[1, 2],
                    help="1 reproduces the first measured run, whose prompt "
                         "described only one side of the trade")
    ap.add_argument("--provenance", default="auto",
                    choices=["auto", "on", "off"],
                    help="auto follows the prompt version (v1 off, v2 on); "
                         "set it explicitly to separate the effect of the "
                         "payload from the effect of the prompt text")
    ap.add_argument("--list-models", action="store_true",
                    help="print the models this key can reach, then exit")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--results", default=str(ROOT / "outputs" / "results.json"))
    ap.add_argument("--out", default=None,
                    help="default: outputs/llm_arm_<model>_p<version>.json")
    args = ap.parse_args()

    load_dotenv()
    provider = args.provider or provider_for(args.model)
    key = api_key_for(provider)

    if args.list_models:
        if not key:
            print(f"set {key_env_names(provider)} first", file=sys.stderr)
            return 2
        for name in list_models(provider, key):
            print(name)
        return 0

    if not key:
        print(f"set {key_env_names(provider)} (or put it in .env) "
              f"to run the {provider} arm", file=sys.stderr)
        return 2

    with_prov = {"auto": None, "on": True, "off": False}[args.provenance]
    prov_in_payload = (args.prompt_version >= 2 if with_prov is None
                       else with_prov)
    tag = "prov" if prov_in_payload else "noprov"
    out_path = Path(args.out) if args.out else (
        ROOT / "outputs" / f"llm_arm_{args.model.replace('.', '-')}"
                           f"_p{args.prompt_version}_{tag}.json")

    results_path = Path(args.results)
    if not results_path.is_file():
        print(f"run run_design.py first: {results_path} is missing",
              file=sys.stderr)
        return 2
    results = json.loads(results_path.read_text(encoding="utf-8"))
    truth_drop = {float(k): bool(v)
                  for k, v in results["truth_boundary"].items()}

    device, noise, cons = DeviceParams(), NoiseModel(), constraints()
    T = device.to_dict()

    runs, log = [], []
    reached = 0
    for rep in range(args.repeats):
        decisions = {}
        for case in breakeven_sweep():
            bel = case.believed(T)
            pool = build_pool(DeviceParams(**bel), cons)
            evidence = screen(DeviceParams(**bel), pool, noise,
                              case.provenance())
            dec = decide(evidence, cons.describe(), bel, case.purpose(),
                         model=args.model, provider=provider,
                         prompt_version=args.prompt_version,
                         with_provenance=with_prov)
            reached += int(dec.used_llm)
            decisions[case.rel_uncertainty] = (
                SUBJECT not in dec.objective.include_params)
            log.append({"repeat": rep, "trust": case.rel_uncertainty,
                        **dec.provenance()})
        runs.append(decisions)

    if reached == 0:
        print("every call fell back to the rule; the agent row is unchanged.",
              file=sys.stderr)
        for entry in log[:1]:
            for e in entry.get("errors", []):
                print(f"  {e}", file=sys.stderr)
        print(f"  if the model name is stale: "
              f"python scripts/run_llm_arm.py --provider {provider} "
              f"--list-models", file=sys.stderr)
        return 3

    levels = sorted(truth_drop)
    print(f"model {args.model}  prompt v{args.prompt_version}  "
          f"provenance {'in payload' if prov_in_payload else 'absent'}  "
          f"{args.repeats} repeats")
    print(f"{'trust':>8}{'truth':>9}{'agent':>24}{'wrong':>7}")
    stability, wrong_total = [], 0
    for t in levels:
        votes = [r[t] for r in runs]
        c = Counter(votes)
        majority = c.most_common(1)[0][0]
        agree = c[majority] / len(votes)
        stability.append(agree)
        wrong = majority != truth_drop[t]
        wrong_total += int(wrong)
        word = "fix" if majority else "estimate"
        print(f"{t * 100:7.0f}%{'fix' if truth_drop[t] else 'estimate':>9}"
              f"{word + f'  ({c[majority]}/{len(votes)})':>24}"
              f"{'  <-' if wrong else '':>7}")

    split = [t for t in levels
             if len(set(r[t] for r in runs)) > 1]
    boundary = max([t for t in levels if runs[0][t]], default=None)
    out = {"model": args.model, "provider": provider,
           "prompt_version": args.prompt_version,
           "provenance_in_payload": prov_in_payload,
           "repeats": args.repeats,
           "calls_reaching_model": reached,
           "decisions": [{str(k): v for k, v in r.items()} for r in runs],
           "wrong": wrong_total, "n_cases": len(levels),
           "mean_self_agreement": sum(stability) / len(stability),
           "levels_not_unanimous": [float(x) for x in split],
           "underpowered": bool(split and args.repeats < 10),
           "boundary_first_run": boundary, "log": log}
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print(f"\nwrong {wrong_total}/{len(levels)};  "
          f"self-agreement {out['mean_self_agreement'] * 100:.0f}%")
    if split:
        pct = ", ".join(f"{t * 100:.0f}%" for t in split)
        print(f"NOT UNANIMOUS at {pct} -- the majority there is a coin flip, "
              f"not a decision")
    if args.repeats < 10 and split:
        print(f"UNDERPOWERED: with {args.repeats} repeats a level the model "
              f"splits on resolves correctly about half the time, so the "
              f"wrong-count carries a swing of a couple of cases on its own. "
              f"Do not compare conditions from this run; use --repeats 15+.")
    print(f"compare with rule_with_provenance: "
          f"{results['boundaries']['rule_with_provenance']['n_wrong']}"
          f"/{len(levels)}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
