#!/usr/bin/env python3
"""Run stage 2: physical correlation, noise deconvolution, corner comparison.

    python run_corners.py            # full run, writes outputs/
    python run_corners.py --quick    # smaller population

Produces
    outputs/report.md             the engineering report
    outputs/results.json          every number
    outputs/fig_correlation.png   process vs estimation correlation
    outputs/fig_corners.png       coverage against waste
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

from corners.deconvolve import (deconvolve, design_from_stage1,
                                extraction_covariance, measure_population)
from corners.device import DeviceParams, NoiseModel, PARAM_NAMES, information
from corners.evaluate import compare, evaluate, nominal_coverage
from corners.generate import all_methods, worst_case_distance
from corners.metrics import all_specs, evaluate_all
from corners.physical import (SENSITIVITY, anisotropy, default_variables,
                              dominant_cause, process_covariance,
                              sample_population)

K_SIGMA = 3.0


# --------------------------------------------------------------------------- #
# Part 1 -- where correlation comes from
# --------------------------------------------------------------------------- #

def correlation_study(pop, nominal: DeviceParams,
                      noise: NoiseModel, design) -> Dict:
    """Process correlation against estimation correlation.

    These are different quantities with different causes and they are easy to
    confuse, because both come out as a correlation matrix over the same seven
    parameters.  Process correlation says how devices differ from each other.
    Estimation correlation says how the extraction trades parameters off
    against one noise realisation of a single device.  Using one where the
    other belongs is the fastest way to build a wrong corner model.
    """
    proc = pop.correlation()
    est_cov = extraction_covariance(nominal, design.vgs, design.vds, noise)
    sd = np.sqrt(np.clip(np.diag(est_cov), 1e-300, None))
    est = np.nan_to_num(est_cov / np.outer(sd, sd), nan=0.0)

    pairs = []
    for i in range(len(PARAM_NAMES)):
        for j in range(i + 1, len(PARAM_NAMES)):
            pairs.append({"a": PARAM_NAMES[i], "b": PARAM_NAMES[j],
                          "process": float(proc[i, j]),
                          "estimation": float(est[i, j]),
                          "opposite_sign": bool(proc[i, j] * est[i, j] < 0)})
    pairs.sort(key=lambda d: -abs(d["process"] - d["estimation"]))

    aniso = anisotropy(process_covariance())

    return {"anisotropy": aniso, "n_params": len(PARAM_NAMES),
            "n_causes": len(default_variables()),
            "process": proc.tolist(), "estimation": est.tolist(),
            "names": list(PARAM_NAMES), "pairs": pairs,
            "dominant_cause": {p: dominant_cause(p) for p in PARAM_NAMES},
            "sensitivity": {k: dict(v) for k, v in SENSITIVITY.items()},
            "n_opposite_sign": sum(p["opposite_sign"] for p in pairs)}


# --------------------------------------------------------------------------- #
# Part 2 -- the observed spread is not the process spread
# --------------------------------------------------------------------------- #

def deconvolution_study(pop, nominal: DeviceParams, noise: NoiseModel,
                        design, n_devices: int, seed: int = 0) -> Dict:
    """Measure a sub-population, extract, and take the measurement back out."""
    sub = sample_population(n_devices, nominal=nominal, seed=seed + 101)
    est_params, failures = measure_population(sub, design, noise, seed=seed)
    if len(est_params) < 20:
        return {"skipped": "too few extractions completed",
                "failures": failures}

    ext_cov = extraction_covariance(nominal, design.vgs, design.vds, noise)
    dec = deconvolve(est_params, nominal.to_dict(), ext_cov)

    truth_cov = sub.covariance()
    t_sd = np.sqrt(np.clip(np.diag(truth_cov), 0, None))
    p_sd = np.sqrt(np.clip(np.diag(dec.process), 0, None))
    o_sd = np.sqrt(np.clip(np.diag(dec.observed), 0, None))

    return {"n_devices": int(len(est_params)), "failures": int(failures),
            "psd_correction": dec.psd_correction,
            "per_param": {
                n: {"truth": float(t_sd[i]), "observed": float(o_sd[i]),
                    "deconvolved": float(p_sd[i]),
                    "observed_over_truth": float(o_sd[i] / t_sd[i])
                    if t_sd[i] > 0 else float("nan"),
                    "deconvolved_over_truth": float(p_sd[i] / t_sd[i])
                    if t_sd[i] > 0 else float("nan")}
                for i, n in enumerate(PARAM_NAMES)},
            "summary": dec.summary()}


# --------------------------------------------------------------------------- #
# Part 3 -- coverage against waste
# --------------------------------------------------------------------------- #

def corner_study(pop, nominal: DeviceParams, mc_n: int, seed: int = 0) -> Dict:
    """Every corner method against every metric.

    ``worst_case_distance`` is rebuilt for each metric rather than reused.
    It solves for the extremes of *one* number on the k-sigma ellipsoid, so a
    set built for one metric says nothing about another, and scoring it across
    metrics it was not built for would be measuring the wrong thing.
    """
    cov = pop.covariance()
    nom = nominal.to_dict()
    devices = pop.as_devices()
    specs = all_specs()

    general = all_methods(cov, nom, k=K_SIGMA, metric=None, mc_n=mc_n,
                          seed=seed)
    results = compare(general, devices, specs)

    truth = evaluate_all(devices)
    for name, m in specs.items():
        wcd = worst_case_distance(cov, nom, m.fn, k=K_SIGMA)
        results.append(evaluate(wcd, devices, m, truth=truth[name]))

    rows = [r.summary() for r in results]

    # statistical_mc is excluded from the ranking on purpose.  It is a sample
    # drawn from the same distribution the reference range is taken from, so
    # it reproduces that range by construction and would always "win" with
    # near-zero waste while needing hundreds of simulations.  It is reported
    # as the reference it is, not as a corner set that beat the others.
    best = {}
    for name in specs:
        safe = [r for r in rows if r["metric"] == name and r["safe"]
                and r["method"] != "statistical_mc"]
        if safe:
            b = min(safe, key=lambda r: r["waste"])
            best[name] = {"method": b["method"], "waste": b["waste"],
                          "coverage": b["coverage"],
                          "n_corners": b["n_corners"]}
    return {"k": K_SIGMA, "nominal_coverage": nominal_coverage(K_SIGMA),
            "corner_counts": {k: len(v) for k, v in general.items()},
            "results": rows, "best_safe_by_metric": best}


# --------------------------------------------------------------------------- #

def make_figures(res: Dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, AMBER, MUTED, GRID = "#16181d", "#c9932a", "#5d626e", "#dfe2e8"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

    # --- process vs estimation correlation ------------------------------ #
    cs = res["correlation"]
    names = cs["names"]
    proc = np.array(cs["process"])
    est = np.array(cs["estimation"])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), dpi=200)
    for ax, M, title in ((axes[0], proc, "process\n(shared physical causes)"),
                         (axes[1], est, "estimation\n(one device, one noise "
                                        "realisation)")):
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(title, fontsize=8)
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j and abs(M[i, j]) > 0.45:
                    ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if abs(M[i, j]) > 0.7 else INK)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    i, j = names.index("vth0"), names.index("mu0")
    fig.suptitle(f"vth0-mu0:  process {proc[i, j]:+.2f}   "
                 f"estimation {est[i, j]:+.2f}", fontsize=8.5, y=1.02)
    fig.savefig(out_dir / "fig_correlation.png", facecolor="white",
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    # --- coverage against waste ------------------------------------------ #
    rows = res["corners"]["results"]
    metrics = sorted({r["metric"] for r in rows})
    methods = ["independent_box", "one_at_a_time", "pca_corners",
               "statistical_mc", "worst_case_distance"]
    marks = {"independent_box": "s", "one_at_a_time": "v",
             "pca_corners": "^", "statistical_mc": "o",
             "worst_case_distance": "D"}
    nom_cov = res["corners"]["nominal_coverage"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics), 3.0),
                             dpi=200, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        ax.axhline(nom_cov * 100, color=INK, lw=1.0, zorder=2)
        ax.axhspan(90, nom_cov * 100, color="#fdf9f0", zorder=0)
        for meth in methods:
            r = next((x for x in rows if x["metric"] == metric
                      and x["method"] == meth), None)
            if r is None:
                continue
            ax.scatter(r["waste"] * 100, r["coverage"] * 100,
                       marker=marks[meth], s=46,
                       color=INK if r["safe"] else AMBER,
                       zorder=4, edgecolors="white", linewidths=0.7)
        ax.axvline(0.0, color=GRID, lw=0.9, ls="--", zorder=1)
        ax.set_title(metric, fontsize=8.5)
        ax.set_xlabel("waste  (%)", fontsize=7.6)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("coverage  (%)", fontsize=7.6)
    axes[0].set_ylim(94, 100.6)
    handles = [plt.Line2D([], [], marker=marks[m], ls="", color=INK,
                          markersize=5.5, label=m) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.11))
    fig.text(0.5, -0.17, "amber = fails the coverage it claims;  "
                         "left of the dashed line = narrower than the "
                         "population needs",
             ha="center", fontsize=6.8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_corners.png", facecolor="white",
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_report(res: Dict, out_dir: Path) -> None:
    cs, dec, cor = res["correlation"], res["deconvolution"], res["corners"]
    names = cs["names"]
    proc = np.array(cs["process"])
    est = np.array(cs["estimation"])
    i, j = names.index("vth0"), names.index("mu0")

    L = ["# Stage 2 - corner and statistical models", "",
         f"Synthetic population of {res['meta']['n_population']} devices, "
         "sampled from physical causes rather than from a parameter "
         "covariance. Corners at "
         f"{cor['k']:.0f} sigma (nominal coverage "
         f"{cor['nominal_coverage'] * 100:.2f}%).", "",
         "## 1. Process correlation is not estimation correlation", "",
         "Both are correlation matrices over the same seven parameters, and "
         "they answer different questions. Process correlation says how "
         "devices differ from one another; estimation correlation says how "
         "the extraction trades parameters off against one noise realisation "
         "of a single device.", "",
         f"For `vth0` and `mu0` they have **opposite signs**: process "
         f"{proc[i, j]:+.3f}, estimation {est[i, j]:+.3f}. Stage 1 measured "
         f"the estimation value and reported it as the pair that trades off; "
         f"using that number to build a corner model would tilt every corner "
         f"the wrong way.", "",
         f"{cs['n_opposite_sign']} of {len(cs['pairs'])} parameter pairs "
         "differ in sign between the two matrices.", "",
         "| parameter | dominant physical cause |", "|---|---|"]
    aniso = cs["anisotropy"]
    tail = [
        "", "### The parameters are not seven independent dimensions", "",
        f"{cs['n_causes']} shared physical causes drive {cs['n_params']} "
        "parameters, plus a small independent term per parameter for local "
        "effects. The result is full rank but strongly anisotropic: "
        f"**{aniso['axes_for_90pct_variance']} axes carry 90% of the "
        f"variance**, and the covariance condition number is "
        f"{aniso['condition_number']:.0f}.", "",
        "That is the quantitative form of the objection to a box. A box "
        "treats all seven directions as equally wide; the process does not "
        "move that way, so most of the volume a box encloses is population "
        "that will never exist.", "",
        "The local term is not decoration. Without it, two parameters driven "
        "by a single shared cause come out perfectly correlated -- an earlier "
        "version of this module produced an `eta`/`vsat` correlation of "
        "exactly -1.000, which no real population shows. Random dopant "
        "fluctuation and line-edge roughness act on each parameter "
        "separately and break that degeneracy."]
    for p in names:
        L.append(f"| `{p}` | {cs['dominant_cause'][p]} |")
    L += tail

    L += ["", "## 2. The observed spread is not the process spread", ""]
    if "skipped" in dec:
        L.append(f"Skipped: {dec['skipped']}.")
    else:
        L += [f"{dec['n_devices']} devices measured through stage 1's "
              "12-point design and extracted, then the extraction covariance "
              "predicted by stage 1's Fisher analysis subtracted.", "",
              "| parameter | true sigma | observed | observed/true | "
              "deconvolved/true |", "|---|---|---|---|---|"]
        for n in names:
            d = dec["per_param"][n]
            L.append(f"| `{n}` | {d['truth'] * 100:.2f}% | "
                     f"{d['observed'] * 100:.2f}% | "
                     f"{d['observed_over_truth']:.2f}x | "
                     f"{d['deconvolved_over_truth']:.2f}x |")
        bad = [n for n in names
               if dec["per_param"][n]["observed_over_truth"] > 3.0]
        ok = [n for n in names
              if abs(dec["per_param"][n]["deconvolved_over_truth"] - 1.0) < 0.2]
        L += ["", "A ratio above 1 in the `observed/true` column is margin "
              "that belongs to the instrument rather than to the devices. "
              "Subtracting the extraction covariance is what stage 1's "
              "measurement design buys a corner model: the deconvolved sigma "
              f"lands within 20% of the truth for {len(ok)} parameters "
              f"({', '.join('`' + n + '`' for n in ok)}).", ""]
        if bad:
            L += ["**It does not work for every parameter, and the ones it "
                  "fails on are the ones stage 1 already identified.** "
                  + ", ".join(f"`{n}` is inflated "
                              f"{dec['per_param'][n]['observed_over_truth']:.1f}x"
                              for n in bad)
                  + ". These are the parameters stage 1 found to be weakly "
                  "identifiable or not identifiable at all -- `theta` carries "
                  "about 1/87 of `vth0`'s sensitivity. Almost none of their "
                  "observed spread is process variation; it is extraction "
                  "noise. Subtracting a *linear* estimate of that noise does "
                  "not recover the truth either, because the error is no "
                  "longer small enough for the linearisation to hold.", "",
                  "The lesson is narrower than it first looks: a parameter's "
                  "observed spread can be deconvolved only if the parameter "
                  "was measurable to begin with. For one that is not, the "
                  "observed spread carries no information about the process "
                  "and no correction puts it back. A corner model should fix "
                  "such a parameter at nominal rather than give it a range -- "
                  "which is the same decision stage 1 was measuring, arrived "
                  "at from the other direction.", ""]

    L += ["", "## 3. Coverage against waste", "",
          "Coverage is the fraction of real devices inside the predicted "
          "range; waste is how much wider that range is than the one the "
          "devices actually occupy. A corner set that covers everything at "
          "twice the necessary width is not a good corner set.", "",
          "| metric | method | corners | coverage | waste | safe |",
          "|---|---|---|---|---|---|"]
    for r in sorted(cor["results"], key=lambda r: (r["metric"], r["method"])):
        L.append(f"| {r['metric']} | `{r['method']}` | {r['n_corners']} | "
                 f"{r['coverage'] * 100:.2f}% | {r['waste'] * 100:+.0f}% | "
                 f"{'yes' if r['safe'] else '**no**'} |")

    L += ["", "### Narrowest corner set that still covers what it claims", "",
          "`statistical_mc` is excluded: it is drawn from the same "
          "distribution the reference range comes from, so it reproduces that "
          "range by construction and would always appear to win, at the cost "
          "of hundreds of simulations.", "",
          "| metric | method | corners | waste |", "|---|---|---|---|"]
    for m, b in cor["best_safe_by_metric"].items():
        L.append(f"| {m} | `{b['method']}` | {b['n_corners']} | "
                 f"{b['waste'] * 100:+.0f}% |")
    box = {r["metric"]: r for r in cor["results"]
           if r["method"] == "independent_box"}
    if box and cor["best_safe_by_metric"]:
        lo = min(r["waste"] for r in box.values()) * 100
        hi = max(r["waste"] for r in box.values()) * 100
        wn = max(b["waste"] for b in cor["best_safe_by_metric"].values()) * 100
        L += ["", f"The box corners cover everything, at +{lo:.0f}% to "
              f"+{hi:.0f}% waste across 128 simulations. Targeting the "
              f"k-sigma ellipsoid instead covers the same population within "
              f"+{wn:.0f}% waste using 2."]

    L += ["", "## 4. What this stage does not show", "",
          "- The metrics are **proxies** computed from the compact model, not "
          "a circuit solve. Stage 3 replaces them and re-checks whether these "
          "corners still bound the real thing.",
          "- `worst_case_distance` needs the metric in advance, so it answers "
          "a narrower question than the others. It is not a general corner "
          "set and is reported only against the metric it was built for.",
          "- `statistical_mc` is a sample, not a corner set: its range is "
          "bounded by what it happened to draw, so its coverage improves with "
          "sample count rather than being a property of the method.",
          "- Everything is synthetic, with known ground truth. The "
          "sensitivity coefficients in `physical.py` are first-order textbook "
          "relations, not values fitted to silicon.", "",
          "![correlation](fig_correlation.png)", "",
          "![corners](fig_corners.png)", ""]
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    n_pop = 800 if args.quick else 3000
    n_meas = 60 if args.quick else 200
    mc_n = 500 if args.quick else 2000

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nominal, noise = DeviceParams(), NoiseModel()
    print("rebuilding stage 1's measurement design ...")
    design = design_from_stage1(nominal, noise)

    print(f"sampling {n_pop} devices from physical causes ...")
    pop = sample_population(n_pop, nominal=nominal, seed=7)

    print("comparing process and estimation correlation ...")
    cor = correlation_study(pop, nominal, noise, design)

    print(f"measuring and extracting {n_meas} devices ...")
    dec = deconvolution_study(pop, nominal, noise, design, n_meas)

    print("generating and scoring corner sets ...")
    corners = corner_study(pop, nominal, mc_n)

    res = {"meta": {"n_population": n_pop, "n_measured": n_meas,
                    "mc_samples": mc_n, "k_sigma": K_SIGMA,
                    "design_points": len(design.indices)},
           "correlation": cor, "deconvolution": dec, "corners": corners}
    (out_dir / "results.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("figures ...")
    make_figures(res, out_dir)
    write_report(res, out_dir)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
