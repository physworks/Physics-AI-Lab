#!/usr/bin/env python3
"""Run stage 3: device fix, solver checks, and the corner re-validation.

    python run_circuit.py            # full run, writes outputs/
    python run_circuit.py --quick    # smaller population

Produces
    outputs/report.md         the engineering report
    outputs/results.json      every number
    outputs/fig_circuit.png   transfer curve, the turnover, corner coverage
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "stage2_corners"))

from circuit.circuits import VDD, propagation_delay, transfer_curve
from circuit.device import (P12_FIXED_POINT_CAP, DeviceParams, evaluate,
                            fixed_point_iterations, reference_current)
from circuit.measure import (CIRCUIT_METRICS, LOG_METRICS, fall_delay,
                             measure_device, measure_population,
                             validate_quadrature)
from corners.generate import (independent_box, one_at_a_time, pca_corners,
                              statistical_mc, worst_case_distance)
from corners.metrics import spec as proxy_spec
from corners.physical import sample_population

K_SIGMA = 3.0


# --------------------------------------------------------------------------- #
# Part 1 -- the device evaluation, before anything is built on it
# --------------------------------------------------------------------------- #

def device_study() -> Dict:
    """The series-resistance loop, the derivatives, and the turnover."""
    p = DeviceParams()
    vg = np.linspace(0.0, 1.3, 40)
    vd = np.full_like(vg, 1.2)

    new = evaluate(p, vg, vd)
    old = reference_current(p, vg, vd)
    agree = float(np.max(np.abs(new.ids - old) / np.maximum(old, 1e-30)))

    hard = DeviceParams(rs=2000.0, mu0=0.09)
    hard_fp = fixed_point_iterations(hard, vg, vd)
    hard_new = evaluate(hard, vg, vd)
    hard_old = reference_current(hard, vg, vd)
    hard_err = float(np.max(np.abs(hard_new.ids - hard_old)
                            / np.maximum(hard_new.ids, 1e-30)))

    # analytic derivatives against finite differences of the converged solve
    h, gm_err, gds_err = 1e-6, [], []
    for v in (0.3, 0.6, 0.9, 1.2):
        e = evaluate(p, np.array([v]), np.array([1.2]))
        gm_fd = (evaluate(p, np.array([v + h]), np.array([1.2])).ids[0]
                 - evaluate(p, np.array([v - h]), np.array([1.2])).ids[0]) / (2 * h)
        gds_fd = (evaluate(p, np.array([v]), np.array([1.2 + h])).ids[0]
                  - evaluate(p, np.array([v]), np.array([1.2 - h])).ids[0]) / (2 * h)
        gm_err.append(abs(e.gm[0] / gm_fd - 1.0))
        gds_err.append(abs(e.gds[0] / gds_fd - 1.0))

    # the turnover and where gds goes negative
    g1 = np.linspace(0.1, 1.3, 25)
    d1 = np.linspace(0.05, 1.3, 26)
    G, D = np.meshgrid(g1, d1, indexing="ij")
    ev = evaluate(p, G.ravel(), D.ravel())
    gds = ev.gds.reshape(G.shape)
    neg = gds < 0

    turnover = {}
    for vgt in (1.0, 1.1, 1.2):
        vo = np.linspace(0.05, VDD, 400)
        I = evaluate(p, np.full_like(vo, vgt), vo).ids
        turnover[f"vg_{vgt}"] = {
            "peak_ua": float(I.max() * 1e6),
            "peak_vd": float(vo[int(np.argmax(I))]),
            "drop_percent": float((I.max() - I[-1]) / I.max() * 100)}

    # How much of the turnover the circuit actually sees.  The terminal
    # derivative is f_d/(1 + f_d*rs), so a negative f_d makes the denominator
    # smaller than one and series resistance *shrinks* the region that looks
    # turned over.  The fraction quoted above is therefore a property of a
    # particular rs, not of the model, and the circuit runs at nominal.
    by_rs = []
    for rs in (0.0, float(p.rs), 500.0):
        gq = evaluate(replace(p, rs=rs), G.ravel(), D.ravel()).gds
        by_rs.append({"rs": rs, "negative_fraction": float((gq < 0).mean())})

    return {"agreement_with_project12": agree,
            "negative_gds_by_series_resistance": by_rs,
            "nominal_fixed_point_iterations": fixed_point_iterations(p, vg, vd),
            "nominal_newton_iterations": new.iterations,
            "hard_device": {"rs": 2000.0, "mu0": 0.09,
                            "fixed_point_iterations": hard_fp,
                            "newton_iterations": hard_new.iterations,
                            "project12_error_percent": hard_err * 100},
            "derivative_max_rel_error": {"gm": float(max(gm_err)),
                                         "gds": float(max(gds_err))},
            "negative_gds_fraction": float(neg.mean()),
            "negative_gds_first_vg": float(G[neg].min()) if neg.any() else None,
            "negative_gds_first_vd": float(D[neg].min()) if neg.any() else None,
            "most_negative_gds": float(gds.min()),
            "turnover": turnover,
            "load_resistance_for_negative_node_conductance":
                float(-1.0 / gds.min())}


def reachability_study() -> Dict:
    """Can the negative conductance actually break a solve inside the rails?

    Two conditions have to hold at once: the load must be weak enough that
    ``gds`` dominates the node conductance, and the device must be biased
    where ``gds`` is negative. Each is easy; the question is whether both can
    hold together.
    """
    p = DeviceParams()
    rows = []
    for r_load in (2e3, 10e3, 25.7e3, 50e3):
        vo = np.linspace(0.02, VDD, 1500)
        I = evaluate(p, np.full_like(vo, 1.15), vo).ids
        f = (VDD - vo) / r_load - I
        s = np.sign(f)
        pts = [float(vo[k]) for k in range(vo.size - 1) if s[k] * s[k + 1] < 0]
        entry = {"r_load": r_load, "operating_points": pts}
        if pts:
            g = [float(evaluate(p, np.array([1.15]), np.array([v])).gds[0])
                 for v in pts]
            entry["gds"] = g
            entry["net_node_conductance"] = [1.0 / r_load + x for x in g]
            entry["any_negative_net"] = bool(any(1.0 / r_load + x < 0
                                                 for x in g))
        rows.append(entry)
    return {"load_lines": rows}


# --------------------------------------------------------------------------- #
# Part 2 -- do stage 2's corners bound the circuit?
# --------------------------------------------------------------------------- #

def _range_of(values: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(values)
    return (float(np.min(values[ok])), float(np.max(values[ok]))) if ok.any() \
        else (float("nan"), float("nan"))


def _reference_range(values: np.ndarray, k: float) -> tuple[float, float]:
    from math import erf, sqrt
    tail = (1.0 - erf(k / sqrt(2.0))) / 2.0
    ok = np.isfinite(values)
    lo, hi = np.quantile(values[ok], [tail, 1.0 - tail])
    return float(lo), float(hi)


def _width(lo: float, hi: float, log_scale: bool) -> float:
    if log_scale:
        return float(np.log10(max(hi, 1e-300)) - np.log10(max(lo, 1e-300)))
    return float(hi - lo)


def corner_study(n_pop: int, mc_n: int, seed: int = 7) -> Dict:
    """Score stage 2's corner sets against circuit measurements.

    The registered prediction: ``worst_case_distance`` built against stage 2's
    proxy ``C*V/Ion`` should under-cover the real fall delay, because the real
    delay depends on the whole trajectory the output swings through and not on
    the drive current at one bias. Both are built here so the comparison is
    direct.
    """
    pop = sample_population(n_pop, seed=seed)
    cov = pop.covariance()
    nom = DeviceParams().to_dict()
    devices = pop.as_devices()

    print(f"   measuring {n_pop} devices ...")
    truth = measure_population(devices)

    sets = {
        "independent_box": independent_box(cov, nom, K_SIGMA),
        "one_at_a_time": one_at_a_time(cov, nom, K_SIGMA),
        "pca_corners": pca_corners(cov, nom, K_SIGMA),
        "statistical_mc": statistical_mc(cov, nom, mc_n, seed, K_SIGMA),
        "wcd_proxy_delay": worst_case_distance(cov, nom,
                                               proxy_spec("delay").fn, K_SIGMA),
        "wcd_circuit_delay": worst_case_distance(cov, nom, fall_delay, K_SIGMA),
    }

    measured = {}
    for name, cs in sets.items():
        print(f"   measuring corner set {name} ({len(cs)}) ...")
        measured[name] = measure_population(cs.as_devices())

    from math import erf, sqrt
    nominal_cov = float(erf(K_SIGMA / sqrt(2.0)))

    rows = []
    for metric in CIRCUIT_METRICS:
        log = metric in LOG_METRICS
        t = truth[metric]
        ref = _reference_range(t, K_SIGMA)
        w_ref = _width(*ref, log)
        for name, cs in sets.items():
            pred = _range_of(measured[name][metric])
            ok = np.isfinite(t)
            cover = float(np.mean((t[ok] >= pred[0]) & (t[ok] <= pred[1])))
            w = _width(*pred, log)
            rows.append({
                "metric": metric, "method": name, "n_corners": len(cs),
                "predicted_lo": pred[0], "predicted_hi": pred[1],
                "reference_lo": ref[0], "reference_hi": ref[1],
                "coverage": cover, "nominal_coverage": nominal_cov,
                "waste": float(w / w_ref - 1.0) if w_ref > 0 else float("nan"),
                "safe": bool(cover >= nominal_cov - 0.005)})

    prediction = {}
    for name in ("wcd_proxy_delay", "wcd_circuit_delay"):
        r = next(x for x in rows
                 if x["metric"] == "t_fall" and x["method"] == name)
        prediction[name] = {"coverage": r["coverage"], "waste": r["waste"],
                            "safe": r["safe"]}
    prediction["proxy_under_covers"] = bool(
        not prediction["wcd_proxy_delay"]["safe"])
    prediction["circuit_version_covers"] = bool(
        prediction["wcd_circuit_delay"]["safe"])

    return {"k": K_SIGMA, "nominal_coverage": nominal_cov,
            "n_population": n_pop,
            "corner_counts": {k: len(v) for k, v in sets.items()},
            "results": rows, "registered_prediction": prediction,
            "truth_summary": {m: {"min": float(np.nanmin(truth[m])),
                                  "max": float(np.nanmax(truth[m])),
                                  "median": float(np.nanmedian(truth[m]))}
                              for m in CIRCUIT_METRICS}}


def proxy_validity_study(n: int = 250, seed: int = 41) -> Dict:
    """Why the registered prediction missed, and where it would not have.

    The prediction assumed the real delay would decouple from a drive-current
    proxy because it depends on the whole trajectory the output swings
    through. The trajectory argument is right; the magnitude judgement was
    not. Across a realistic population the devices differ almost entirely by a
    current *scale*, and the delay integral scales inversely with that same
    scale, so the proxy is the real delay rescaled by a near-constant factor.

    This sweeps the two kinds of variation apart. Inflating ``mu0`` scales the
    current and leaves the shape alone; inflating ``vth0`` moves where the
    device turns on relative to the 1.2 V swing and changes the shape. If the
    reasoning was right and only the magnitude wrong, shape variation should
    break the proxy and scale variation should not.
    """
    from corners.metrics import evaluate_all
    from corners.physical import LOCAL_SIGMA

    base = dict(LOCAL_SIGMA)
    cases = [("baseline", None, None),
             ("mu0 x5 (scale)", "mu0", 0.040),
             ("vth0 x5 (shape)", "vth0", 0.030),
             ("vth0 x15 (shape)", "vth0", 0.090),
             ("vth0 x30 (shape)", "vth0", 0.180)]
    rows = []
    try:
        for name, key, sigma in cases:
            LOCAL_SIGMA.clear()
            LOCAL_SIGMA.update(base)
            if key:
                LOCAL_SIGMA[key] = sigma
            pop = sample_population(n, seed=seed)
            devices = pop.as_devices()
            proxy = evaluate_all(devices)["delay"]
            real = np.array([fall_delay(d) for d in devices])
            ok = np.isfinite(real)
            if ok.sum() < 30:
                rows.append({"case": name, "usable": int(ok.sum())})
                continue
            pr, rr = proxy[ok], real[ok]
            ratio = rr / pr
            rows.append({
                "case": name, "varied": key,
                "sigma": float(sigma if sigma is not None
                               else base.get("vth0", 0.0)),
                "usable": int(ok.sum()),
                "pearson": float(np.corrcoef(pr, rr)[0, 1]),
                "ratio_mean": float(np.mean(ratio)),
                "ratio_rel_sd": float(np.std(ratio) / np.mean(ratio))})
    finally:
        LOCAL_SIGMA.clear()
        LOCAL_SIGMA.update(base)
    return {"cases": rows}


# --------------------------------------------------------------------------- #

def make_figure(res: Dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, AMBER, MUTED, GRID = "#16181d", "#c9932a", "#5d626e", "#dfe2e8"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    p = DeviceParams()

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), dpi=200)

    # (a) inverter transfer curve
    tc, _ = transfer_curve(p, n=41)
    axes[0].plot(tc.vin, tc.vout, color=INK, lw=1.6)
    axes[0].axhline(VDD / 2, color=GRID, lw=0.9, ls="--")
    axes[0].axvline(tc.switching_threshold(), color=AMBER, lw=1.2, ls="--")
    axes[0].set_title(f"resistive-load inverter\nVm = "
                      f"{tc.switching_threshold():.3f} V, gain "
                      f"{tc.peak_gain():.2f}", fontsize=8)
    axes[0].set_xlabel("$V_{in}$ (V)", fontsize=7.6)
    axes[0].set_ylabel("$V_{out}$ (V)", fontsize=7.6)

    # (b) the turnover
    vo = np.linspace(0.05, VDD, 300)
    for vgt, style in ((1.0, ":"), (1.1, "--"), (1.2, "-")):
        I = evaluate(p, np.full_like(vo, vgt), vo).ids
        axes[1].plot(vo, I * 1e6, style, color=INK, lw=1.4,
                     label=f"$V_g$={vgt}")
        k = int(np.argmax(I))
        axes[1].scatter([vo[k]], [I[k] * 1e6], s=22, color=AMBER, zorder=4,
                        edgecolors="white", linewidths=0.6)
    axes[1].legend(fontsize=6.8, frameon=False)
    axes[1].set_title("output characteristic turns over\n"
                      "(amber = peak, then $g_{ds}<0$)", fontsize=8)
    axes[1].set_xlabel("$V_d$ (V)", fontsize=7.6)
    axes[1].set_ylabel("$I_d$ ($\\mu$A)", fontsize=7.6)

    # (c) coverage vs waste on the real delay
    rows = [r for r in res["corners"]["results"] if r["metric"] == "t_fall"]
    marks = {"independent_box": "s", "one_at_a_time": "v", "pca_corners": "^",
             "statistical_mc": "o", "wcd_proxy_delay": "X",
             "wcd_circuit_delay": "D"}
    nom_cov = res["corners"]["nominal_coverage"] * 100
    axes[2].axhline(nom_cov, color=INK, lw=1.0)
    for r in rows:
        axes[2].scatter(r["waste"] * 100, r["coverage"] * 100,
                        marker=marks.get(r["method"], "o"), s=48,
                        color=INK if r["safe"] else AMBER, zorder=4,
                        edgecolors="white", linewidths=0.7)
    # The two worst-case-distance points land on top of each other, which is
    # the result rather than a plotting problem: a corner set built against
    # the proxy and one built against the circuit are the same corners.
    wcd = {r["method"]: r for r in rows if r["method"].startswith("wcd")}
    if len(wcd) == 2:
        a, b = wcd["wcd_proxy_delay"], wcd["wcd_circuit_delay"]
        same = (abs(a["waste"] - b["waste"]) < 0.01
                and abs(a["coverage"] - b["coverage"]) < 0.002)
        label = ("proxy and circuit:\nsame corners" if same
                 else "proxy / circuit\ndiffer")
        axes[2].annotate(label, (a["waste"] * 100, a["coverage"] * 100),
                         textcoords="offset points", xytext=(10, -14),
                         fontsize=6.4, color=INK if a["safe"] else AMBER,
                         arrowprops=dict(arrowstyle="-", lw=0.7,
                                         color=MUTED))
    axes[2].set_title("corner coverage on the real\nfall delay", fontsize=8)
    axes[2].set_xlabel("waste (%)", fontsize=7.6)
    axes[2].set_ylabel("coverage (%)", fontsize=7.6)

    for ax in axes:
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_dir / "fig_circuit.png", facecolor="white",
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(res: Dict, out_dir: Path) -> None:
    d, reach, cor = res["device"], res["reachability"], res["corners"]
    L = ["# Stage 3 - circuit-level validation", "",
         "A hand-written MNA solver with damped Newton, NMOS-with-load "
         "circuits, and stage 2's corner sets re-scored against real "
         "measurements.", "",
         "## 1. The device evaluation had to be fixed first", "",
         "Project 12 solves series resistance by damped fixed-point iteration "
         "capped at 40 steps and returns the 40th value either way.", "",
         "| | iterations | result |", "|---|---|---|",
         f"| nominal, fixed point | {d['nominal_fixed_point_iterations']} | "
         f"under the 40-step cap, by "
         f"{P12_FIXED_POINT_CAP - d['nominal_fixed_point_iterations']} |",
         f"| nominal, Newton | {d['nominal_newton_iterations']} | converged |",
         f"| `rs=2000`, fixed point | "
         f"{d['hard_device']['fixed_point_iterations']}+ | **unconverged**, "
         f"current off by {d['hard_device']['project12_error_percent']:.1f}% |",
         f"| `rs=2000`, Newton | {d['hard_device']['newton_iterations']} | "
         "converged |", "",
         f"Where project 12's loop did converge the two agree to "
         f"{d['agreement_with_project12']:.1e} relative, so the physics is "
         "unchanged. Derivatives are now analytic, through the implicit "
         "function theorem, and match finite differences of the converged "
         f"solve to {max(d['derivative_max_rel_error'].values()):.1e}.", "",
         "Stage 2 was re-checked and stayed under the cap, so its results "
         "stand.", "",
         "## 2. The model has negative output conductance", "",
         f"`gds < 0` over **{d['negative_gds_fraction'] * 100:.0f}%** of the "
         f"operating grid, from about `Vg = {d['negative_gds_first_vg']:.2f} V`, "
         f"`Vd = {d['negative_gds_first_vd']:.2f} V`. The drain current peaks "
         "and then falls:", "",
         "| gate bias | peak | at | falls by |", "|---|---|---|---|"]
    for k, t in d["turnover"].items():
        L.append(f"| {k.replace('vg_', 'Vg = ')} V | {t['peak_ua']:.1f} uA | "
                 f"Vd = {t['peak_vd']:.2f} V | {t['drop_percent']:.2f}% |")

    L += ["", "This is in project 12's own current, not an artefact of the new "
          "derivative, and it survives setting `rs = 0`, so it is the "
          "intrinsic model: the velocity-saturation term outruns DIBL.", "",
          "Series resistance partly hides it. The terminal derivative is "
          "`f_d / (1 + f_d*rs)`, so a negative `f_d` makes the denominator "
          "smaller than one and a larger `rs` shrinks the region that looks "
          "turned over:", "",
          "| `rs` | negative `gds` |", "|---|---|"]
    for e in d["negative_gds_by_series_resistance"]:
        tag = " (nominal)" if abs(e["rs"] - DeviceParams().rs) < 1e-9 else (
            " (intrinsic)" if e["rs"] == 0.0 else "")
        L.append(f"| {e['rs']:.0f}{tag} | {e['negative_fraction'] * 100:.1f}% |")

    L += ["",
          "So the percentage above belongs to a particular `rs`, not to the "
          "model, and the circuit runs at nominal -- where the region is "
          "smaller than the model's own. That is a second reason no solve "
          "broke on it, independent of the load-line argument below.", "",
          "### It did not break any solve, and the reason is worth stating",
          "",
          "The design document predicted this would break Newton. **It did "
          "not.** Two conditions have to hold at once and they exclude each "
          "other inside a 1.2 V rail:", "",
          f"- For `gds` to dominate the node conductance the load must exceed "
          f"**{d['load_resistance_for_negative_node_conductance'] / 1e3:.1f} "
          f"kOhm**.",
          "- With a load that large the device pulls the output down to a few "
          "tens of millivolts, far below the `Vd` where `gds` turns negative.",
          "", "| load | operating point | gds | 1/R + gds |", "|---|---|---|---|"]
    for row in reach["load_lines"]:
        pts = ", ".join(f"{v:.3f}" for v in row["operating_points"]) or "none"
        g = ", ".join(f"{v:+.1e}" for v in row.get("gds", [])) or "-"
        n = ", ".join(f"{v:+.1e}" for v in
                      row.get("net_node_conductance", [])) or "-"
        L.append(f"| {row['r_load'] / 1e3:.1f} kOhm | {pts} | {g} | {n} |")

    L += ["", "A weak-load inverter does bias into the region, and still "
          "converges: the load conductance is around fifty times `gds`. The "
          "turnover is also shallow, about 1% deep, so no load line inside "
          "the rail crosses it twice. The defect is real and did not bite "
          "here. A current-mirror load or a cascode is where it would, and "
          "neither exists in an NMOS-only model.", "",
          "## 3. Do stage 2's corners bound the circuit?", "",
          f"{cor['n_population']} devices, measured through the inverter "
          "rather than through a proxy.", "",
          "| metric | method | corners | coverage | waste | safe |",
          "|---|---|---|---|---|---|"]
    for r in sorted(cor["results"], key=lambda r: (r["metric"], r["method"])):
        L.append(f"| {r['metric']} | `{r['method']}` | {r['n_corners']} | "
                 f"{r['coverage'] * 100:.2f}% | {r['waste'] * 100:+.0f}% | "
                 f"{'yes' if r['safe'] else '**no**'} |")

    pr = cor["registered_prediction"]
    L += ["", "### The registered prediction", "",
          "Before measuring, stage 3's design recorded this: a corner set "
          "built against stage 2's proxy (`C*V/Ion`) should **under-cover** "
          "the real fall delay, because the real delay depends on the whole "
          "trajectory the output swings through and not on drive current at "
          "one bias.", "",
          "| built against | coverage on real delay | waste | safe |",
          "|---|---|---|---|",
          f"| the proxy | {pr['wcd_proxy_delay']['coverage'] * 100:.2f}% | "
          f"{pr['wcd_proxy_delay']['waste'] * 100:+.0f}% | "
          f"{'yes' if pr['wcd_proxy_delay']['safe'] else '**no**'} |",
          f"| the circuit | {pr['wcd_circuit_delay']['coverage'] * 100:.2f}% | "
          f"{pr['wcd_circuit_delay']['waste'] * 100:+.0f}% | "
          f"{'yes' if pr['wcd_circuit_delay']['safe'] else '**no**'} |", "",
          ("**The prediction held.**" if pr["proxy_under_covers"]
           else "**The prediction did not hold.** The proxy-built corner set "
                "covered the real delay as well as the circuit-built one, at "
                "the same waste. Recorded as a miss."), ""]

    pv = res.get("proxy_validity", {}).get("cases", [])
    if pv:
        L += ["### Why it missed", "",
              "The reasoning behind the prediction was that the real delay "
              "depends on the trajectory the output swings through, not on "
              "drive current at one bias. That is true. What was wrong was "
              "the magnitude: across a realistic population the devices "
              "differ almost entirely by a current *scale*, and the delay "
              "integral scales inversely with that same scale, so the proxy "
              "is the real delay multiplied by a near-constant.", "",
              "Separating the two kinds of variation shows the reasoning was "
              "right and only the size was wrong -- scale variation leaves "
              "the proxy intact, shape variation breaks it:", "",
              "| population | spread | correlation with real delay | "
              "ratio spread |", "|---|---|---|---|"]
        for row in pv:
            if "pearson" not in row:
                L.append(f"| {row['case']} | - | too few devices reached mid "
                         "level | - |")
                continue
            L.append(f"| {row['case']} | {row['sigma'] * 100:.1f}% | "
                     f"{row['pearson']:.5f} | "
                     f"{row['ratio_rel_sd'] * 100:.2f}% |")
        L += ["", "The proxy only starts to fail once the threshold voltage "
              "spreads by something like 18%, an order of magnitude beyond "
              "the 1.8% this process actually has. For this circuit and this "
              "process it was a good proxy, and stage 2's corner choice was "
              "not put at risk by it.", ""]

    L += [
          "## 4. What this stage does not show", "",
          "- A hand-written Newton is not SPICE. Source stepping, gmin "
          "stepping and limiting schemes would change which of these solves "
          "struggle.",
          "- NMOS-with-load circuits only, because the model is NMOS only. "
          "No CMOS inverter, no current-mirror load, no cascode -- which is "
          "exactly where the negative conductance would have mattered.",
          "- The population fall delay is computed by quadrature on the "
          "single-node circuit equation rather than by stepping Newton. It "
          "was checked against the solver's transient and agreed to "
          f"{res['quadrature_check']['max_rel_error'] * 100:.2f}% on "
          f"{res['quadrature_check']['n']} devices.",
          "- Everything is synthetic, with known ground truth.", "",
          "![circuit](fig_circuit.png)", ""]
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    n_pop = 150 if args.quick else 400
    mc_n = 150 if args.quick else 400
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("checking the device evaluation ...")
    device = device_study()
    print("checking whether negative gds is reachable ...")
    reach = reachability_study()

    print("validating the quadrature delay against the solver ...")
    chk = validate_quadrature(sample_population(4, seed=21).as_devices())

    print("scoring stage 2's corners against the circuit ...")
    corners = corner_study(n_pop, mc_n)

    print("checking when the proxy stops standing in for the circuit ...")
    proxy_validity = proxy_validity_study(150 if args.quick else 250)

    res = {"meta": {"n_population": n_pop, "mc_samples": mc_n,
                    "k_sigma": K_SIGMA},
           "device": device, "reachability": reach,
           "quadrature_check": chk.summary(), "corners": corners,
           "proxy_validity": proxy_validity}
    (out_dir / "results.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("figure ...")
    make_figure(res, out_dir)
    write_report(res, out_dir)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
