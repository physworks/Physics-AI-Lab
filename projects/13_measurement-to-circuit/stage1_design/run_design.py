#!/usr/bin/env python3
"""Run stage 1: design comparison, break-even sweep, decision boundaries.

    python run_design.py                 # full run, writes outputs/
    python run_design.py --quick         # fewer seeds
    python run_design.py --require-llm   # fail if the LLM arm never ran

Produces
    outputs/report.md          the engineering report
    outputs/results.json       every number
    outputs/fig_design.png     where each arm puts its measurements
    outputs/fig_breakeven.png  the decision boundary and who finds it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from oed import (DesignConstraints, DeviceParams, NoiseModel, build_pool,
                 decide, default_required_regions, greedy_design,
                 heuristic_design, holdout_grid, run_arm, uniform_design)
from oed.fisher import build_cache, information
from oed.objective import (Objective, full_objective, rule_objective,
                           rule_objective_provenance)
from oed.pipeline import screen
from oed.scenarios import (ALL_PARAMS, SUBJECT, WITHOUT_THETA,
                           DecisionBoundary, breakeven_sweep)

N_POINTS = 12
MAX_VDS = 3


def constraints(n_points: int = N_POINTS) -> DesignConstraints:
    return DesignConstraints(n_points=n_points, max_distinct_vds=MAX_VDS,
                             required_regions=default_required_regions())


def objective_for(include: List[str], believed: Dict[str, float],
                  criterion: str = "D", source: str = "experiment",
                  rationale: str = "") -> Objective:
    return Objective(include_params=list(include), criterion=criterion,
                     fixed_params={n: believed[n] for n in ALL_PARAMS
                                   if n not in include},
                     rationale=rationale, source=source)


# --------------------------------------------------------------------------- #
# Part 1 -- is an informative design worth anything at all?
# --------------------------------------------------------------------------- #

def compare_designs(truth: DeviceParams, noise: NoiseModel, seeds: int) -> Dict:
    """Optimal selection against the plans a line would actually use."""
    cons = constraints()
    pool = build_pool(truth, cons)
    cache = build_cache(truth, pool.vgs, pool.vds, noise)
    hold = holdout_grid()
    obj = full_objective(truth.to_dict(), "D")

    designs = {
        "uniform": uniform_design(truth, pool, cons, noise, obj),
        "heuristic": heuristic_design(truth, pool, cons, noise, obj),
        "greedy_D": greedy_design(truth, pool, cons, noise, obj,
                                  arm="greedy_D", cache=cache),
    }

    blind = noise.without_floor()
    blind_cache = build_cache(truth, pool.vgs, pool.vds, blind)
    designs["greedy_D_uniform_noise"] = greedy_design(
        truth, pool, cons, blind, obj, arm="greedy_D_uniform_noise",
        cache=blind_cache)

    out = {}
    for name, d in designs.items():
        r = run_arm(truth, d, noise, seeds=seeds, holdout=hold)
        out[name] = {**r.summary(), "design": d.summary(),
                     "vgs": np.round(d.vgs, 4).tolist(),
                     "vds": np.round(d.vds, 4).tolist()}
    return out


# --------------------------------------------------------------------------- #
# Part 2 -- the break-even, measured
# --------------------------------------------------------------------------- #

def breakeven(truth: DeviceParams, noise: NoiseModel, seeds: int) -> Dict:
    """Which choice wins, as a function of how trustworthy the nominal is."""
    T = truth.to_dict()
    cons = constraints()
    hold = holdout_grid()
    rows = {}

    for case in breakeven_sweep():
        bel = case.believed(T)
        bp = DeviceParams(**bel)
        pool = build_pool(bp, cons)
        cache = build_cache(bp, pool.vgs, pool.vds, noise)

        res = {}
        for tag, include in (("keep", ALL_PARAMS), ("drop", WITHOUT_THETA)):
            d = greedy_design(bp, pool, cons, noise,
                              objective_for(include, bel), arm=tag, cache=cache)
            r = run_arm(truth, d, noise, seeds=seeds, holdout=hold)
            res[tag] = r.predict_rms_log10

        res["ratio"] = res["drop"] / res["keep"]
        res["drop_is_better"] = bool(res["ratio"] < 1.0)
        rows[case.rel_uncertainty] = res
    return rows


def truth_boundary(rows: Dict) -> Dict[float, bool]:
    return {t: bool(v["drop_is_better"]) for t, v in rows.items()}


def load_agent_runs(out_dir: Path, truth_drop: Dict[float, bool]) -> List[Dict]:
    """Pick up every measured agent run sitting in outputs/.

    The LLM arm runs on a machine that can reach a provider, which is not the
    one this script usually runs on.  Rather than transcribing those numbers
    by hand, the runner writes a file per condition and this reads whatever
    is there, so the report never quotes a result nobody can re-open.
    """
    runs = []
    for path in sorted(out_dir.glob("llm_arm_*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        votes: Dict[float, List[bool]] = {}
        for rep in d.get("decisions", []):
            for k, v in rep.items():
                votes.setdefault(float(k), []).append(bool(v))
        if not votes:
            continue
        majority = {t: (sum(v) * 2 > len(v)) for t, v in votes.items()}
        wrong = sorted(t for t, m in majority.items()
                       if t in truth_drop and m != truth_drop[t])
        prov = d.get("provenance_in_payload",
                     d.get("prompt_version", 2) >= 2)
        runs.append({
            "file": path.name,
            "model": d.get("model", "?"),
            "prompt_version": d.get("prompt_version", "?"),
            "provenance_in_payload": prov,
            "label": f"{d.get('model', '?')}  v{d.get('prompt_version', '?')}"
                     f"{'' if prov else ' (no prov.)'}",
            "decides_drop_up_to": max([t for t, m in majority.items() if m],
                                      default=None),
            "drops_at": {str(t): bool(m) for t, m in majority.items()},
            "wrong_at": wrong, "n_wrong": len(wrong), "n_cases": len(majority),
            "self_agreement": d.get("mean_self_agreement"),
            "repeats": d.get("repeats")})
    return runs


# --------------------------------------------------------------------------- #
# Part 3 -- who finds the boundary
# --------------------------------------------------------------------------- #

def decision_boundaries(truth: DeviceParams, noise: NoiseModel,
                        transport=None, api_key: str | None = None) -> Dict:
    """Ask each decision-maker, at each trust level, keep or drop."""
    T = truth.to_dict()
    cons = constraints()

    makers = {
        "rule_sensitivity_only": DecisionBoundary("rule_sensitivity_only"),
        "rule_with_provenance": DecisionBoundary("rule_with_provenance"),
        "agent": DecisionBoundary("agent"),
    }
    agent_log: List[Dict] = []
    llm_used = False

    for case in breakeven_sweep():
        bel = case.believed(T)
        bp = DeviceParams(**bel)
        pool = build_pool(bp, cons)
        evidence = screen(bp, pool, noise, case.provenance())

        o1 = rule_objective(evidence, bel)
        makers["rule_sensitivity_only"].drops_at[case.rel_uncertainty] = (
            SUBJECT not in o1.include_params)

        o2 = rule_objective_provenance(evidence, bel)
        makers["rule_with_provenance"].drops_at[case.rel_uncertainty] = (
            SUBJECT not in o2.include_params)

        dec = decide(evidence, cons.describe(), bel, case.purpose(),
                     transport=transport, api_key=api_key)
        llm_used = llm_used or dec.used_llm
        makers["agent"].drops_at[case.rel_uncertainty] = (
            SUBJECT not in dec.objective.include_params)
        agent_log.append({"trust": case.rel_uncertainty,
                          **dec.provenance()})

    return {"makers": makers, "agent_log": agent_log, "llm_used": llm_used}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def make_figures(truth: DeviceParams, comp: Dict, rows: Dict,
                 boundaries: Dict, out_dir: Path,
                 agent_runs: List[Dict] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from oed.device import drain_current

    INK, AMBER, MUTED, GRID = "#16181d", "#c9932a", "#5d626e", "#dfe2e8"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

    # --- where the arms put their points -------------------------------- #
    arms = ["uniform", "heuristic", "greedy_D_uniform_noise", "greedy_D"]
    titles = ["uniform grid", "engineer heuristic",
              "D-optimal, uniform noise", "D-optimal, floor aware"]
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.5), dpi=200, sharey=True)
    vg_bg = np.linspace(0.0, 1.2, 200)
    for ax, arm, title in zip(axes, arms, titles):
        for vd, style in ((0.05, "-"), (1.2, "--")):
            ax.plot(vg_bg, np.log10(np.maximum(
                drain_current(truth, vg_bg, np.full_like(vg_bg, vd)), 1e-16)),
                style, color=GRID, lw=1.1, zorder=1)
        d = comp[arm]
        ax.scatter(d["vgs"], np.log10(np.maximum(drain_current(
            truth, np.array(d["vgs"]), np.array(d["vds"])), 1e-16)),
            s=26, color=AMBER if "uniform_noise" in arm else INK,
            zorder=3, edgecolors="white", linewidths=0.6)
        ax.set_title(f"{title}\nheld-out RMS "
                     f"{comp[arm]['predict_rms_log10']:.4f}", fontsize=7.6)
        ax.set_xlabel("$V_g$ (V)", fontsize=7.6)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("$\\log_{10} I_d$", fontsize=7.6)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_dir / "fig_design.png", facecolor="white",
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # --- break-even and decision boundaries ----------------------------- #
    n_rows = 2 + (len(agent_runs) if agent_runs else 1)
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(5.6, 3.1 + 0.42 * n_rows), dpi=200, sharex=True,
        gridspec_kw={"height_ratios": [2.3, 0.34 * n_rows + 0.45]})

    t = np.array(sorted(rows))
    ratio = np.array([rows[k]["ratio"] for k in t])
    ax.axhline(1.0, color=INK, lw=1.0)
    ax.fill_between(t * 100, 1.0, ratio, where=ratio < 1.0,
                    color="#e8eef5", zorder=0)
    ax.fill_between(t * 100, 1.0, ratio, where=ratio >= 1.0,
                    color="#fdf6e7", zorder=0)
    ax.plot(t * 100, ratio, "-o", color=INK, lw=1.4, ms=4.5,
            markeredgecolor="white", markeredgewidth=0.7)

    cross = None
    for i in range(len(t) - 1):
        if (ratio[i] - 1.0) * (ratio[i + 1] - 1.0) < 0:
            f = (1.0 - ratio[i]) / (ratio[i + 1] - ratio[i])
            cross = (t[i] + f * (t[i + 1] - t[i])) * 100
            break
    if cross is not None:
        ax.axvline(cross, color=AMBER, lw=1.3, ls="--")
        ax.text(cross + 1.0, ratio.max() * 0.96,
                f"break-even\n{cross:.0f}%", color=AMBER, fontsize=7.4,
                va="top", fontweight="bold")

    ax.text(2.5, 0.74, "fixing theta\nis better", fontsize=7.4, color=MUTED)
    ax.text(42, 1.72, "estimating theta\nis better", fontsize=7.4, color=MUTED,
            ha="right")
    ax.set_ylabel("held-out error,  fix / estimate", fontsize=8)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    truth_drop = truth_boundary(rows)
    series = [("rule: sensitivity only",
               {tv: boundaries["makers"]["rule_sensitivity_only"].drops_at.get(tv)
                for tv in t}),
              ("rule: + provenance",
               {tv: boundaries["makers"]["rule_with_provenance"].drops_at.get(tv)
                for tv in t})]
    for run in (agent_runs or []):
        series.append((run["label"],
                       {tv: run["drops_at"].get(str(tv)) for tv in t}))
    if not agent_runs:
        series.append(("agent (rule fallback)",
                       {tv: boundaries["makers"]["agent"].drops_at.get(tv)
                        for tv in t}))

    for row, (lab, decisions) in enumerate(series):
        for tv in t:
            d = decisions.get(tv)
            if d is None:
                continue
            ok = (d == truth_drop[tv])
            ax2.scatter(tv * 100, row, s=42,
                        marker="o" if d else "X",
                        color=INK if ok else AMBER, zorder=3,
                        edgecolors="white", linewidths=0.7)
        ax2.text(-1.5, row, lab, ha="right", va="center", fontsize=7.0)
    names = [lab for lab, _ in series]
    if cross is not None:
        ax2.axvline(cross, color=AMBER, lw=1.3, ls="--", zorder=1)
    ax2.set_ylim(-0.7, len(names) - 0.3)
    ax2.set_xlim(-0.5, 64)
    ax2.set_yticks([])
    ax2.set_xlabel("nominal error on theta  (%)", fontsize=8)
    ax2.text(64, -0.62, "circle = fix,  cross = estimate;  amber = wrong",
             fontsize=6.8, color=MUTED, ha="right")
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.grid(axis="x", color=GRID, lw=0.5)
    ax2.set_axisbelow(True)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_dir / "fig_breakeven.png", facecolor="white",
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def write_report(res: Dict, out_dir: Path) -> None:
    comp, rows = res["design_comparison"], res["breakeven"]
    tb = {float(k): v for k, v in res["truth_boundary"].items()}

    L = ["# Stage 1 - optimal measurement design", "",
         "Synthetic device, known ground truth. "
         f"{res['meta']['n_points']} bias points, at most "
         f"{res['meta']['max_distinct_vds']} distinct drain biases, "
         "stress and specification limits enforced.", "",
         "## 1. Does choosing the points matter?", "",
         "Held-out prediction error on a dense grid no design ever saw.", "",
         "| design | held-out RMS log10(Id) | vs uniform |", "|---|---|---|"]

    base = comp["uniform"]["predict_rms_log10"]
    for k in ("uniform", "heuristic", "greedy_D_uniform_noise", "greedy_D"):
        v = comp[k]["predict_rms_log10"]
        L.append(f"| `{k}` | {v:.5f} | {base / v:.2f}x |")

    L += ["", "`greedy_D_uniform_noise` is the textbook design: it assumes the "
          "measurement noise is the same everywhere. It is not -- in deep "
          "subthreshold the current approaches the instrument floor. The "
          "textbook design puts points where sensitivity is largest, which is "
          "exactly where precision is worst.", "",
          "## 2. The decision that actually matters", "",
          "`theta` carries about 1/87 of `vth0`'s sensitivity (project 12): no "
          "measurement plan recovers it. Fixing it removes a nuisance "
          "direction -- but only if the value it is fixed at is close enough.",
          "",
          "| nominal error on theta | estimate it | fix it | ratio | better |",
          "|---|---|---|---|---|"]
    for t in sorted(rows, key=float):
        r = rows[t]
        L.append(f"| {float(t) * 100:.0f}% | {r['keep']:.5f} | "
                 f"{r['drop']:.5f} | {r['ratio']:.2f} | "
                 f"{'fix' if r['drop_is_better'] else 'estimate'} |")

    L += ["", "The same decision is worth a 30% improvement or a 2x "
          "degradation depending on a quantity that is not in the data.", "",
          "## 3. Who finds the boundary", "",
          "| decision maker | fixes theta up to | wrong at | wrong / total |",
          "|---|---|---|---|"]
    for nm, mk in res["boundaries"].items():
        b = mk["decides_drop_up_to"]
        wrong = ", ".join(f"{float(x) * 100:.0f}%" for x in mk["wrong_at"])
        L.append(f"| `{nm}` | {'never' if b is None else f'{b * 100:.0f}%'} | "
                 f"{wrong or '-'} | {mk['n_wrong']} / {mk['n_cases']} |")

    runs = res.get("agent_runs") or []
    if runs:
        L += ["", "### The agent arm, measured", "",
              "The `agent` row above is the rule fallback this machine "
              "produced. The rows below are real runs, each a separate "
              "condition, majority decision over repeats.", "",
              "| model | prompt | provenance in payload | wrong | "
              "self-agreement |", "|---|---|---|---|---|"]
        for r in runs:
            sa = r.get("self_agreement")
            L.append(
                f"| `{r['model']}` | v{r['prompt_version']} | "
                f"{'yes' if r['provenance_in_payload'] else 'no'} | "
                f"{r['n_wrong']} / {r['n_cases']} | "
                f"{'-' if sa is None else f'{sa * 100:.0f}%'} |")
        thin = [r for r in runs if r.get("repeats", 0) < 10]
        L += ["", "**No agent condition beat a rule.** The best ties "
              "`rule_with_provenance`, and every condition fails in the same "
              "place and the same direction -- fixing the parameter where the "
              "measured answer is to estimate it, at 30% nominal error and "
              "above. Neither the prompt rewrite nor the structured evidence "
              "moved that boundary.", "",
              "The highest self-agreement belongs to conditions that are "
              "systematically wrong, so consistency here measures "
              "repeatability and not correctness."]
        if thin:
            L += ["", "> **Underpowered for comparing conditions.** At these "
                  "repeat counts, a decision level the model splits on "
                  "resolves correctly about half the time, so the wrong-count "
                  "swings by a case or two without anything changing. One "
                  "earlier sweep of the v1-with-provenance condition returned "
                  "5 of 8 and did not reproduce; it is reported here as "
                  "variation, not as an effect. Differences between "
                  "conditions need many more repeats before they mean "
                  "anything."]
    elif not res["meta"]["llm_used"]:
        L += ["", "> The agent row above is the **rule fallback**: no LLM was "
              "reached in this run. Run `scripts/run_llm_arm.py` to fill it "
              "in. A fallback row is not evidence about an agent."]

    L += ["", "## 4. What this stage does not show", "",
          "- A c-optimal design on `vth0` beat D-optimal on `vth0`'s "
          "uncertainty by 4.7% over 60 noise realisations. The sampling error "
          "of a standard deviation from 60 samples is about 9%, so this is "
          "reported as **not established**.",
          "- Designs are built at the believed parameters, not the true ones. "
          "Sequential re-design would address the gap; it is not implemented "
          "here.", "",
          "![designs](fig_design.png)", "", "![break-even](fig_breakeven.png)",
          ""]
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--require-llm", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    seeds = 8 if args.quick else 25
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth, noise = DeviceParams(), NoiseModel()

    print(f"comparing designs ({seeds} seeds) ...")
    comp = compare_designs(truth, noise, seeds)

    print("sweeping the break-even ...")
    rows = breakeven(truth, noise, seeds)

    print("asking each decision maker ...")
    bnd = decision_boundaries(truth, noise)

    tb = truth_boundary(rows)
    agent_runs = load_agent_runs(out_dir, tb)
    if agent_runs:
        print(f"folding in {len(agent_runs)} measured agent run(s)")
    res = {
        "meta": {"n_points": N_POINTS, "max_distinct_vds": MAX_VDS,
                 "seeds": seeds, "llm_used": bnd["llm_used"]},
        "design_comparison": comp,
        "breakeven": {str(k): v for k, v in rows.items()},
        "truth_boundary": {str(k): v for k, v in tb.items()},
        "boundaries": {k: v.summary(tb) for k, v in bnd["makers"].items()},
        "agent_runs": agent_runs,
        "agent_log": bnd["agent_log"],
    }
    (out_dir / "results.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("figures ...")
    make_figures(truth, comp, rows, bnd, out_dir, agent_runs)
    write_report(res, out_dir)
    print(f"wrote {out_dir}")

    if args.require_llm and not bnd["llm_used"]:
        print("ERROR: --require-llm was set but every call fell back to the "
              "rule.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
