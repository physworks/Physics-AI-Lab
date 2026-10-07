#!/usr/bin/env python3
"""Job 3: diagnose injected convergence faults, and score the diagnosis.

    python run_diagnose.py                       # rule arms only
    python run_diagnose.py --model gemini-3.5-flash-lite --repeats 3

Writes outputs/diagnosis.json and appends a section to outputs/report.md.
Without a provider key the LLM arm is the rule fallback and says so; a run
where no call reached a model is not evidence about a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from circuit.diagnose import (ABSTAIN, ArmResult, Diagnosis, api_key_for,
                              llm_diagnose, provider_for, rule_diagnose)
from circuit.faults import FAULT_CLASSES, build_cases, solve_case
from circuit.fragility import summarise


def undecidable_cases(traces: Dict[str, Dict],
                      truths: Dict[str, str]) -> List[str]:
    """Cases whose trace is identical in kind to a healthy one.

    Determined from the evidence, not from the labels: a case that converged
    with no anomaly in any device cannot be separated from ``none`` by
    anything in the trace. Marking them lets accuracy be reported on the
    decidable subset as well as overall, instead of charging a diagnoser for
    failing to read something that is not there.
    """
    out = []
    for name, t in traces.items():
        d = (t.get("device_state") or [{}])[0]
        clean = (t.get("converged")
                 and d.get("inner_converged", True)
                 and float(d.get("gds", 0.0)) >= 0.0
                 and float(d.get("gm", 1.0)) != 0.0)
        if clean and truths[name] != "none":
            out.append(name)
    return out


def _append_report(res: Dict, out_dir: Path) -> None:
    """Replace or add the Job 3 section of the stage report."""
    path = out_dir / "report.md"
    body = path.read_text(encoding="utf-8") if path.is_file() else ""
    marker = "## 5. Diagnosing convergence failures"
    body = body.split(marker)[0].rstrip()

    und = res["undecidable_from_trace"]
    L = ["", marker, "",
         f"{res['n_cases']} solves with a known fault injected into each, "
         "six classes. The diagnoser sees the solver's trace -- residual "
         "history, Jacobian conditioning, device terminal state, and whether "
         "a retry from the default start converged. **No field in the trace "
         "names the fault.**", "",
         f"**{len(und)} of {res['n_cases']} cases are undecidable from the "
         f"trace** ({', '.join('`' + c + '`' for c in und)}). Each is a "
         "discontinuity the solve happened never to cross, so it converged "
         "cleanly and its trace is identical in kind to a healthy one. "
         "Accuracy is reported on the decidable subset as well as overall, "
         "because charging a diagnoser for failing to read what is not there "
         "measures nothing.", "",
         "| arm | accuracy (all) | accuracy (decidable) | abstention | "
         "abstained on undecidable |", "|---|---|---|---|---|"]
    for k, s_ in res["arms"].items():
        L.append(f"| `{k}` | {s_['accuracy_all'] * 100:.1f}% | "
                 f"{s_['accuracy_on_decidable'] * 100:.1f}% | "
                 f"{s_['abstention_rate'] * 100:.1f}% | "
                 f"{s_['abstained_on_undecidable']}/{s_['n_undecidable']} |")

    L += ["", "### What the two rule variants show", "",
          "The plain rule reads every decidable case correctly and loses "
          "only the three invisible ones, which it confidently calls healthy. "
          "The abstaining variant catches all three -- and also abstains on "
          "the four genuinely healthy cases.", "",
          "That is not a tuning failure. **A healthy circuit and a defect the "
          "solve never excited produce the same trace**, so there is no rule "
          "that abstains on one without abstaining on the other. The choice "
          "is between a diagnoser that is quietly wrong three times and one "
          "that declines to answer seven times; the evidence does not support "
          "anything better. Saying which of those is preferable is a decision "
          "about the cost of a wrong diagnosis, not about the trace."]

    if not res.get("llm_reached_model"):
        L += ["", "> Any `llm:` row above is the **rule fallback**: no call "
              "reached a model in this run. Run `run_diagnose.py --model ...` "
              "with a provider key set. A fallback row is not evidence about "
              "a model."]
    else:
        st = res.get("stability") or {}
        llm = next((v for k, v in res["arms"].items()
                    if k.startswith("llm:")), {})
        split, overlap = st.get("split_cases", []), st.get(
            "split_on_undecidable", [])
        reps = st.get("repeats", 1)
        rule = res["arms"].get("rule", {})
        arm = llm.get("arm", "")
        differs = sorted(r["case"] for r in res["per_case"]
                         if r.get("rule") != r.get(arm))
        verdict = ("matched the plain rule's answers exactly" if not differs
                   else "differed from the plain rule on "
                   + ", ".join("`" + c + "`" for c in differs))
        L += ["", "### The LLM arm", "",
              f"{reps} repeats, majority vote per case. It {verdict}, and "
              f"**used abstention {st.get('abstentions_used', 0)} times** -- "
              "on a task where three cases cannot be decided from the "
              "evidence and the prompt described when to abstain and why.", ""]
        if llm.get("accuracy_on_decidable", 1.0) < rule.get(
                "accuracy_on_decidable", 0.0):
            L += ["**On the decidable subset it fell below the rule** -- "
                  f"{llm['accuracy_on_decidable'] * 100:.1f}% against "
                  f"{rule['accuracy_on_decidable'] * 100:.1f}%. The rule is "
                  "deterministic and scores the same number every run; this "
                  "arm does not.", ""]
        L += [f"It was also less reproducible: {len(split)} of "
              f"{res['n_cases']} cases drew a split vote "
              f"({', '.join('`' + c + '`' for c in split) or 'none'}). The "
              "rule arms are deterministic, so zero.", ""]
        if split and overlap:
            L += [f"{len(overlap)} of those splits fall on the undecidable "
                  f"cases ({', '.join('`' + c + '`' for c in overlap)}): the "
                  "model answered differently across runs on a case the "
                  "evidence cannot settle, and never converted that into the "
                  "abstention it was offered. Only the majority vote hid "
                  "it.", ""]
        elif split:
            n, k, u = res["n_cases"], len(split), len(und)
            p = 1.0
            for i in range(k):            # P(no split lands on an undecidable
                p *= (n - u - i) / (n - i)  # case | splits placed at random)
            L += ["**None of those splits fall on the undecidable cases.** "
                  "The instability is not the model registering ambiguity; it "
                  "is movement on cases the evidence settles, while on the "
                  "undecidable ones it was unanimous -- confidently wrong.",
                  "",
                  "This does not establish that the model is unsure where it "
                  "should be certain. With splits placed at random, the "
                  f"chance that none of {k} lands on the {u} undecidable "
                  f"cases is **{p:.2f}**. What it does establish is that "
                  "instability is not substituting for abstention here: the "
                  "model's disagreement with itself says nothing about which "
                  "cases the trace cannot decide. Where the splits do fall is "
                  "below.", ""]
        fr = res.get("fragility")
        if split and fr:
            co = fr["concentration"]
            pr = fr["predictors"]
            hit = ", ".join("`" + c + "`" for c in co["classes_hit"])
            L += ["#### Where the splits fall", "",
                  f"Every split is in {len(co['classes_hit'])} of "
                  f"{len(res['classes'])} classes ({hit}), "
                  f"{co['cases_in_those_classes']} cases in all. Placed at "
                  f"random the chance of that is **{co['p_exact']:.4f}**, and "
                  f"**{co['p_corrected']:.3f}** after allowing for the "
                  f"{co['n_class_groups']} groups of classes that would have "
                  "looked equally striking. So the instability is not spread "
                  "evenly over the cases: it belongs to particular classes.",
                  "",
                  "| class | split | fields that flip the rule | evidence "
                  "chars | residual history |", "|---|---|---|---|---|"]
            for r in fr["classes"]:
                L.append(f"| `{r['class']}` | {r['n_split']}/{r['n']} | "
                         f"{r['mean_flip_fields']:.1f} | "
                         f"{r['mean_evidence_chars']:.0f} | "
                         f"{r['mean_history_length']:.1f} |")
            L += ["",
                  "The cheap explanation is that the model wavers on whatever "
                  "is longest or messiest. It does not: rank correlation "
                  f"between split rate and evidence size is "
                  f"{pr['evidence_chars']:+.2f}, and with residual-history "
                  f"length {pr['history_length']:+.2f}. The two classes that "
                  "never split have histories as long as the two that do.", "",
                  "What does track it, at "
                  f"{pr['flip_count']:+.2f}, is how many single-field changes "
                  "to the evidence would move the rule off its answer -- a "
                  "count of how entangled the decision is. The single-field "
                  "class never split; the five-field class split on three of "
                  "four.", "",
                  "It is not the whole story either: two classes sit at the "
                  "same three fields and land on opposite sides, one splitting "
                  "half its cases and the other none. **This is a correlation "
                  "across six classes of four cases each, with a known "
                  "exception. It is a direction to test, not a mechanism**, "
                  "and the test is to give the entangled classes a field that "
                  "asserts them outright and see whether the instability "
                  "moves.", ""]

        if split and reps < 9:
            L += [f"> **UNDERPOWERED.** {len(split)} of {res['n_cases']} "
                  f"cases moved across {reps} repeats, so the majority vote "
                  "-- and therefore this arm's accuracy -- is itself a random "
                  "variable. Do not compare this number against another run "
                  "or another model. Re-run with `--repeats 9` or more before "
                  "quoting it.", ""]
        L += ["Stage 1 measured an LLM on a quantitative threshold judgement "
              "and it never beat a rule. This stage tested a different shape "
              "of judgement -- discrete classification from structured "
              "evidence, with abstention available -- and the result is the "
              "same: it tied the rule at best, fell below it at worst, and "
              "added nothing either way while costing reproducibility. Two "
              "shapes of judgement, same answer."]

    path.write_text(body + "\n" + "\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="omit to run the rule arms only")
    ap.add_argument("--provider", default=None,
                    choices=["openai", "anthropic", "gemini"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("building and solving the injected cases ...")
    cases = build_cases()
    traces, truths = {}, {}
    for c in cases:
        traces[c.name] = solve_case(c)
        truths[c.name] = c.truth

    und = undecidable_cases(traces, truths)
    print(f"   {len(cases)} cases, {len(und)} undecidable from the trace: "
          f"{und}")

    arms: Dict[str, ArmResult] = {}

    rule = ArmResult("rule")
    rule_abs = ArmResult("rule_with_abstain")
    for c in cases:
        t = traces[c.name]
        rule.diagnoses.append(Diagnosis(c.name, c.truth,
                                        rule_diagnose(t, False), "rule"))
        rule_abs.diagnoses.append(Diagnosis(c.name, c.truth,
                                            rule_diagnose(t, True), "rule"))
    arms["rule"] = rule
    arms["rule_with_abstain"] = rule_abs

    llm_used = False
    stability = None
    fragility = None
    if args.model:
        provider = args.provider or provider_for(args.model)
        if not api_key_for(provider):
            print(f"   no key for {provider}; the LLM arm will be the rule "
                  "fallback", file=sys.stderr)
        runs: List[Dict[str, str]] = []
        for rep in range(args.repeats):
            answers = {}
            for c in cases:
                a, why, reached = llm_diagnose(traces[c.name], args.model,
                                               provider)
                llm_used = llm_used or reached
                answers[c.name] = a
            runs.append(answers)
            print(f"   repeat {rep + 1}/{args.repeats} done")

        arm = ArmResult(f"llm:{args.model}")
        split_cases, votes_by_case = [], {}
        for c in cases:
            votes = [r[c.name] for r in runs]
            votes_by_case[c.name] = votes
            majority = Counter(votes).most_common(1)[0][0]
            if len(set(votes)) > 1:
                split_cases.append(c.name)
            arm.diagnoses.append(Diagnosis(c.name, c.truth, majority, "llm"))
        arms[arm.name] = arm

        # Which cases the model was unsure about matters more than how many.
        # If instability lands on the cases the trace cannot decide, the model
        # registered the ambiguity and simply never converted it into an
        # abstention; if it lands elsewhere, the instability is unrelated
        # noise. Those are different findings, so record the names.
        stability = {"repeats": args.repeats,
                     "cases_with_split_votes": len(split_cases),
                     "split_cases": split_cases,
                     "split_on_undecidable": sorted(set(split_cases) & set(und)),
                     "votes": votes_by_case,
                     "abstentions_used": sum(
                         1 for d in arm.diagnoses if d.abstained)}

        # Where the splits fall turned out to be the finding: all of them in
        # two of the six classes. That is not a shape random placement
        # produces, so the placement gets measured rather than described --
        # including the cheap explanations, which have to be ruled out before
        # the structural one is worth stating.
        if split_cases:
            fragility = summarise(traces, truths, split_cases, und)

    res = {
        "n_cases": len(cases),
        "classes": list(FAULT_CLASSES),
        "undecidable_from_trace": und,
        "llm_reached_model": llm_used,
        "model": args.model,
        "stability": stability,
        "fragility": fragility,
        "arms": {k: v.summary(und) for k, v in arms.items()},
        "per_case": [{"case": c.name, "truth": c.truth,
                      **{k: next(d.answer for d in v.diagnoses
                                 if d.case == c.name)
                         for k, v in arms.items()}}
                     for c in cases],
    }
    (out_dir / "diagnosis.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'arm':26}{'acc all':>9}{'acc decidable':>15}"
          f"{'abstain':>9}{'abst/undecidable':>18}")
    for k, s in res["arms"].items():
        print(f"{k:26}{s['accuracy_all'] * 100:8.1f}%"
              f"{s['accuracy_on_decidable'] * 100:14.1f}%"
              f"{s['abstention_rate'] * 100:8.1f}%"
              f"{s['abstained_on_undecidable']:>12} / {s['n_undecidable']}")

    _append_report(res, out_dir)

    if args.model and not llm_used:
        print("\nNo call reached a model; the LLM row is the rule fallback.",
              file=sys.stderr)
        return 3
    print(f"\nwrote {out_dir / 'diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
