#!/usr/bin/env python3
"""Build the TCAD surrogate artefact consumed by ``SurrogateDeviceModel``.

Input: a CSV of TCAD sweep results, one row per simulated device, with columns

    <param_1> ... <param_k>   fractional deviations from nominal
    vth                       V
    ion                       A/um
    ioff                      A/um

Output: ``artifacts/tcad_surrogate.npz`` holding a linear-plus-quadratic
response surface for [vth, log10 ion, log10 ioff].

The surface is intentionally simple.  It is a *plug-in point*, not the modelling
contribution: swapping it for a neural surrogate means changing this script and
``SurrogateDeviceModel.evaluate`` only, because everything else in the pipeline
talks to the ``DeviceModel`` interface.

``--demo`` fits the surface against the analytic model instead of real TCAD
data.  That exercises the ``model_agreement`` check end to end, and it is
labelled in the artefact so it can never be mistaken for a TCAD-trained model.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from wafer2device.s3_device.models import AnalyticDeviceModel

DEFAULT_PARAMS = ["gate_oxide_thickness", "channel_doping", "gate_length",
                  "interface_trap_density", "series_resistance"]


def fit(X: np.ndarray, Y: np.ndarray):
    """Least-squares fit of y = bias + sum_i (w1_i x_i + w2_i x_i^2)."""
    k = X.shape[1]
    A = np.hstack([np.ones((len(X), 1)), X, X ** 2])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    bias = coef[0]
    w1 = coef[1:1 + k]
    w2 = coef[1 + k:1 + 2 * k]
    resid = Y - A @ coef
    rmse = np.sqrt((resid ** 2).mean(axis=0))
    return bias, w1, w2, rmse


def load_csv(path: str, params: List[str]):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{path} is empty")
    missing = [c for c in params + ["vth", "ion", "ioff"] if c not in rows[0]]
    if missing:
        raise SystemExit(f"missing columns in {path}: {missing}")
    X = np.array([[float(r[p]) for p in params] for r in rows])
    Y = np.column_stack([
        np.array([float(r["vth"]) for r in rows]),
        np.log10(np.array([float(r["ion"]) for r in rows])),
        np.log10(np.array([float(r["ioff"]) for r in rows])),
    ])
    return X, Y


def demo_samples(params: List[str], n: int = 600, seed: int = 11):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-0.10, 0.10, size=(n, len(params)))
    m = AnalyticDeviceModel()
    vth, ion, ioff = [], [], []
    for row in X:
        dev = {p: np.array([v]) for p, v in zip(params, row)}
        out = m.evaluate(dev)
        vth.append(out["vth"][0]); ion.append(out["ion"][0]); ioff.append(out["ioff"][0])
    Y = np.column_stack([np.array(vth), np.log10(ion), np.log10(ioff)])
    return X, Y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="TCAD sweep table")
    ap.add_argument("--demo", action="store_true",
                    help="fit against the analytic model (NOT TCAD data)")
    ap.add_argument("--params", nargs="*", default=DEFAULT_PARAMS)
    ap.add_argument("--out", default=str(ROOT / "artifacts/tcad_surrogate.npz"))
    args = ap.parse_args()

    if args.demo:
        X, Y = demo_samples(args.params)
        provenance = "DEMO: fitted to the analytic model, not TCAD data"
    elif args.csv:
        X, Y = load_csv(args.csv, args.params)
        provenance = f"fitted to TCAD sweeps from {Path(args.csv).name}"
    else:
        raise SystemExit("pass --csv <file> or --demo")

    bias, w1, w2, rmse = fit(X, Y)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, params=np.array(args.params), w1=w1, w2=w2, bias=bias,
             provenance=provenance, n_samples=len(X))

    print(f"samples      : {len(X)}")
    print(f"provenance   : {provenance}")
    print(f"RMSE vth     : {rmse[0]:.5f} V")
    print(f"RMSE log10Ion: {rmse[1]:.5f}")
    print(f"RMSE log10Ioff: {rmse[2]:.5f}")
    print(f"written      : {out}")
    print("\nEnable it with  device_model.secondary: surrogate  in configs/pipeline.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
