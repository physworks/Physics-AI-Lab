#!/usr/bin/env python3
"""Render a four-panel figure for one case.

    observed wafer map | radial profile | implied stress field | Vth before/after

The middle two panels are the ones worth looking at: they put the *observed*
defect map and the *assumed* variation field side by side, which is how the
pattern-consistency check decides whether a hypothesis is telling a story the
wafer actually supports.

    python scripts/plot_case.py --case edge_ring_plus_center_00 --provider stub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wafer2device.gate import load_yaml
from wafer2device.orchestrator import Pipeline
from wafer2device.s1_pattern.data import generate_dataset
from wafer2device.s3_device.models import variation_fields


def plot(case, model, spec, out_path: Path) -> Path:
    wmap = case.wafer_map.value
    n = wmap.shape[0]
    masked = np.ma.masked_where(wmap < 0, wmap)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

    axes[0].imshow(masked, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[0].set_title(f"observed map ({case.wafer_map.provenance.value})")
    axes[0].axis("off")

    d = case.descriptors.value
    centers = np.linspace(0, 1, len(d.radial_profile))
    axes[1].plot(centers, d.radial_profile, "o-")
    axes[1].set_xlabel("normalised radius")
    axes[1].set_ylabel("defect rate")
    axes[1].set_title(f"radial profile (slope {d.radial_slope:+.3f})")
    axes[1].grid(alpha=0.3)

    h = case.hypothesis.value if case.hypothesis else None
    stress = None
    if h is not None:
        fields = variation_fields(h, n)
        stress = np.zeros((n, n))
        for f in fields.values():
            stress += np.clip(f, 0.0, None)

    if stress is not None and float(stress.max()) > 0.0:
        axes[2].imshow(np.ma.masked_where(wmap < 0, stress), cmap="magma")
        terms = ", ".join(f"{t.param.split('_')[0]}/{t.spatial_form}" for t in h.terms)
        axes[2].set_title(f"assumed stress\n{terms}", fontsize=9)
    else:
        # A flat field would render as a featureless block and read as a bug.
        # Say why there is nothing to draw instead.
        msg = ("null hypothesis:\nno variation proposed" if h is not None
               else "escalated:\nno approved hypothesis")
        axes[2].text(0.5, 0.5, msg, ha="center", va="center", fontsize=9,
                     color="#555")
        axes[2].set_title("assumed stress", fontsize=9)
    axes[2].axis("off")

    if h is not None and case.compensation is not None:
        fields = variation_fields(h, n)
        base = model.evaluate(fields)["vth"]
        opt = model.evaluate(fields, case.compensation.value.knobs)["vth"]
        lo = spec["vth_nominal"] - spec["vth_tolerance"]
        hi = spec["vth_nominal"] + spec["vth_tolerance"]
        inside = (wmap >= 0)
        axes[3].hist(base[inside], bins=40, alpha=0.6, label="before")
        axes[3].hist(opt[inside], bins=40, alpha=0.6, label="after")
        axes[3].axvline(lo, color="k", ls="--", lw=1)
        axes[3].axvline(hi, color="k", ls="--", lw=1)
        axes[3].set_xlabel("Vth (V)")
        axes[3].set_title(f"out-of-spec "
                          f"{case.compensation.value.baseline_out_of_spec:.2f} -> "
                          f"{case.compensation.value.optimized_out_of_spec:.2f}",
                          fontsize=9)
        axes[3].legend(fontsize=8)
    else:
        axes[3].text(0.5, 0.5, "compensation not run", ha="center", va="center")
        axes[3].axis("off")

    fig.suptitle(f"{case.case_id}   |   gate: "
                 f"{case.approval.decision if case.approval else '-'}   |   "
                 f"source: {h.source if h else '-'}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="edge_ring_plus_center_00")
    ap.add_argument("--provider", choices=["rule", "stub", "cache", "openai"],
                    default="rule")
    ap.add_argument("--approval", default="auto")
    ap.add_argument("--out-dir", default="outputs/figures")
    args = ap.parse_args()

    cfg = load_yaml(str(ROOT / "configs/pipeline.yaml"))
    checks_cfg = load_yaml(str(ROOT / "configs/checks.yaml"))
    wl_cfg = load_yaml(str(ROOT / "configs/whitelist.yaml"))
    cfg["approval"]["mode"] = args.approval
    if args.provider == "rule":
        cfg["hypothesis"]["mode"] = "rule"
    else:
        cfg["hypothesis"]["mode"] = "llm"
        cfg["hypothesis"]["llm"]["provider"] = args.provider

    pipe = Pipeline(cfg, checks_cfg, wl_cfg)
    cases = {c.case_id: c for c in generate_dataset(
        n_per_scenario=2, n=int(cfg["data"]["grid"]), seed=int(cfg["data"]["seed"]))}
    if args.case not in cases:
        print(f"unknown case; available: {sorted(cases)[:8]} ...")
        return 2

    case = pipe.run_case(cases[args.case])
    path = plot(case, pipe.model, checks_cfg["device_spec"],
                ROOT / args.out_dir / f"{args.case}_{args.provider}.png")
    print(f"written -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
