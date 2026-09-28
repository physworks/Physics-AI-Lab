#!/usr/bin/env python3
"""Run the extraction, the identifiability analysis, and the validation.

    python run_extraction.py              # full run, writes outputs/
    python run_extraction.py --quick      # fewer seeds, for a fast check

Produces:
    outputs/report.md              the engineering report
    outputs/results.json           every number, for re-use
    outputs/fig_iv.png             measured vs fitted I-V
    outputs/fig_identifiability.png sensitivity, correlation, noise sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cmext.extraction import (extract_simultaneous, extract_staged,
                              recovery_error)
from cmext.identifiability import analyse, first_failure, noise_sweep
from cmext.model import DeviceParams, drain_current, measure, sweep_grid

NOISE = 0.01
FLOOR = 1e-12


def monte_carlo(truth, vg, vd, extractor, n_seeds: int, noise=NOISE):
    """Empirical spread of the recovered parameters across noise realisations."""
    names = DeviceParams.names()
    ests = {n: [] for n in names}
    escalated = 0
    for s in range(n_seeds):
        ids = measure(truth, vg, vd, noise_rel=noise, seed=s)
        r = extractor(vg, vd, ids)
        escalated += int(r.escalated)
        for n, v in r.params.to_dict().items():
            ests[n].append(v)
    out = {}
    for n in names:
        a = np.array(ests[n], dtype=float)
        out[n] = {"mean": float(a.mean()), "std": float(a.std()),
                  "rel_std": float(a.std() / max(abs(a.mean()), 1e-12)),
                  "rel_err": float(abs(a.mean() - truth.to_dict()[n])
                                   / max(abs(truth.to_dict()[n]), 1e-12))}
    return out, escalated


def make_figures(truth, est, vg, vd, ids, ident, sweep, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---------------- I-V fit ----------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for vdv, c in zip(sorted(set(vd)), ["#1f4e79", "#a8761f", "#8b2f2f"]):
        m = vd == vdv
        o = np.argsort(vg[m])
        ax[0].semilogy(vg[m][o], ids[m][o], "o", ms=3, color=c, alpha=.55,
                       label=f"meas Vds={vdv}")
        ax[0].semilogy(vg[m][o], drain_current(est, vg[m], vdv * np.ones(m.sum()))[o],
                       "-", color=c, lw=1.4)
        ax[1].plot(vg[m][o], ids[m][o] * 1e3, "o", ms=3, color=c, alpha=.55)
        ax[1].plot(vg[m][o], drain_current(est, vg[m], vdv * np.ones(m.sum()))[o] * 1e3,
                   "-", color=c, lw=1.4)
    ax[0].set_xlabel("Vgs (V)"); ax[0].set_ylabel("Id (A)")
    ax[0].set_title("log scale — subthreshold", fontsize=10)
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].set_xlabel("Vgs (V)"); ax[1].set_ylabel("Id (mA)")
    ax[1].set_title("linear scale — strong inversion", fontsize=10)
    ax[1].grid(alpha=.3)
    fig.suptitle("Measured (points) vs extracted model (lines)", fontsize=11)
    fig.tight_layout(); fig.savefig(out_dir / "fig_iv.png", dpi=130); plt.close(fig)

    # ---------------- identifiability ----------------
    names = ident.names
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    order = np.argsort([ident.sensitivity[n] for n in names])
    ax[0].barh([names[i] for i in order],
               [ident.sensitivity[names[i]] for i in order], color="#1f4e79")
    ax[0].set_xscale("log"); ax[0].set_xlabel("RMS Δlog10(Id) per 1% change")
    ax[0].set_title("Sensitivity — what the data can see", fontsize=10)
    ax[0].grid(alpha=.3, axis="x")

    im = ax[1].imshow(ident.correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=45, fontsize=8)
    ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            v = ident.correlation[i, j]
            if i != j and abs(v) >= 0.9:
                ax[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                           fontsize=7, color="white" if abs(v) > .6 else "black")
    ax[1].set_title("Parameter correlation — what trades off", fontsize=10)
    fig.colorbar(im, ax=ax[1], fraction=.046)

    lv = np.array(sweep["noise_levels"]) * 100
    for n in names:
        ax[2].plot(lv, np.array(sweep["per_param"][n]["mean_err"]) * 100,
                   "o-", ms=3, lw=1.2, label=n)
    ax[2].axhline(10, color="k", ls="--", lw=1)
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("measurement noise (%)"); ax[2].set_ylabel("recovery error (%)")
    ax[2].set_title("Noise sweep — what breaks first", fontsize=10)
    ax[2].legend(fontsize=7, ncol=2); ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_identifiability.png", dpi=130); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    n_seeds = 5 if args.quick else args.seeds
    sweep_seeds = 4 if args.quick else 8

    out = ROOT / "outputs"; out.mkdir(exist_ok=True)

    truth = DeviceParams()
    vg, vd = sweep_grid()
    ids = measure(truth, vg, vd, noise_rel=NOISE, seed=0)

    print("running staged extraction ...")
    staged = extract_staged(vg, vd, ids, noise_floor=FLOOR)
    print("running simultaneous extraction ...")
    simul = extract_simultaneous(vg, vd, ids, noise_floor=FLOOR)

    print(f"monte carlo ({n_seeds} seeds) ...")
    mc_staged, esc_staged = monte_carlo(truth, vg, vd,
                                        lambda a, b, c: extract_staged(a, b, c, FLOOR),
                                        n_seeds)
    mc_simul, esc_simul = monte_carlo(truth, vg, vd,
                                      lambda a, b, c: extract_simultaneous(a, b, c, FLOOR),
                                      n_seeds)

    print("identifiability analysis ...")
    ident = analyse(truth, vg, vd, noise_rel=NOISE)

    print(f"noise sweep ({sweep_seeds} seeds per level) ...")
    sw = noise_sweep(truth, vg, vd,
                     lambda a, b, c, **k: extract_simultaneous(a, b, c, FLOOR),
                     n_seeds=sweep_seeds)
    breaks = first_failure(sw)

    make_figures(truth, simul.params, vg, vd, ids, ident, sw, out)

    results = {
        "truth": truth.to_dict(),
        "staged": staged.to_dict(),
        "simultaneous": simul.to_dict(),
        "recovery_staged": recovery_error(truth, staged.params),
        "recovery_simultaneous": recovery_error(truth, simul.params),
        "monte_carlo": {"n_seeds": n_seeds,
                        "staged": mc_staged, "staged_escalated": esc_staged,
                        "simultaneous": mc_simul, "simultaneous_escalated": esc_simul},
        "identifiability": ident.to_dict(),
        "noise_sweep": sw, "breaks_at": breaks,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    write_report(results, out / "report.md")
    print(f"\nwritten -> {out}/report.md, results.json, fig_iv.png, fig_identifiability.png")
    return 0


def write_report(r: dict, path: Path) -> None:
    t = r["truth"]; ident = r["identifiability"]; mc = r["monte_carlo"]
    L = ["# Extraction report", "",
         "Synthetic device, known ground truth, 1% relative measurement noise.",
         "", "## 1. Recovery against ground truth", "",
         "| parameter | truth | staged | error | simultaneous | error |",
         "|---|---|---|---|---|---|"]
    for n in ident["names"]:
        L.append(f"| `{n}` | {t[n]:.5g} | {r['staged']['params'][n]:.5g} | "
                 f"{r['recovery_staged'][n]*100:.1f}% | "
                 f"{r['simultaneous']['params'][n]:.5g} | "
                 f"{r['recovery_simultaneous'][n]*100:.1f}% |")
    L += ["", f"Staged run escalated: **{r['staged']['escalated']}** "
              f"({mc['staged_escalated']}/{mc['n_seeds']} across seeds)  ",
          f"Simultaneous run escalated: **{r['simultaneous']['escalated']}** "
          f"({mc['simultaneous_escalated']}/{mc['n_seeds']} across seeds)", ""]

    L += ["## 2. Does the uncertainty analysis predict the real spread?", "",
          "The Jacobian gives a predicted 1-sigma uncertainty per parameter "
          "without running any Monte Carlo. Comparing it against the measured "
          "spread over independent noise realisations is what makes the "
          "analysis usable rather than decorative.", "",
          "| parameter | predicted 1σ | measured spread | sensitivity |",
          "|---|---|---|---|"]
    for n in ident["names"]:
        L.append(f"| `{n}` | {ident['rel_std_error'][n]*100:.2f}% | "
                 f"{mc['simultaneous'][n]['rel_std']*100:.2f}% | "
                 f"{ident['sensitivity'][n]:.5f} |")
    L += ["", f"Condition number of JᵀJ: **{ident['condition_number']:.2e}**", ""]

    L += ["## 3. What trades off against what", ""]
    if ident["strong_pairs"]:
        L.append("| pair | correlation |"); L.append("|---|---|")
        for a, b, c in ident["strong_pairs"]:
            L.append(f"| `{a}` ↔ `{b}` | {c:+.3f} |")
    else:
        L.append("No pair exceeded |r| = 0.90.")
    L += ["", "## 4. Where each parameter stops being meaningful", "",
          "Noise level at which mean recovery error first exceeds 10%.", "",
          "| parameter | breaks at | error at 10% noise |", "|---|---|---|"]
    for n, d in r["breaks_at"].items():
        b = "—" if d["breaks_at"] is None else f"{d['breaks_at']*100:.1f}% noise"
        L.append(f"| `{n}` | {b} | {d['err_at_max_noise']*100:.1f}% |")
    L += ["", "![I-V](fig_iv.png)", "", "![identifiability](fig_identifiability.png)", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
