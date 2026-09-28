"""Wafer map sources.

Two sources are supported and they are *never* silently mixed:

* ``synthetic`` -- maps generated from a known process-variation ground truth.
  Tagged ``Provenance.SYNTHETIC``.  Because the ground truth is known, these
  cases are what the rule-vs-LLM comparison experiment scores against.
* ``wm811k``    -- real WM-811K wafer maps (Wu et al.), loaded from a local
  ``LSWMD.pkl``.  Tagged ``Provenance.MEASURED``.  Real maps have *no* process
  ground truth, which is exactly why the hypothesis stage is explicitly an
  assumption and not a measurement.

The dataset is not bundled: drop ``LSWMD.pkl`` into ``data/`` and set
``data.source: wm811k`` in ``configs/pipeline.yaml``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..contracts import Provenance, Tagged, WaferCase


def wafer_grid(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (r, theta, mask) on an n x n grid inscribed in the unit circle."""
    ax = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(ax, ax)
    r = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)
    mask = r <= 1.0
    return r, theta, mask


def spatial_field(form: str, r: np.ndarray, theta: np.ndarray,
                  magnitude: float, angle_deg: float = 0.0) -> np.ndarray:
    """Evaluate a whitelisted spatial form on the wafer grid.

    Shared by the data generator and the device stage so that "the shape the
    hypothesis claims" and "the shape the wafer shows" are directly comparable.
    """
    if form == "uniform":
        shape = np.ones_like(r)
    elif form == "radial_linear":
        shape = r
    elif form == "radial_quadratic":
        shape = r ** 2
    elif form == "edge_ring":
        shape = 1.0 / (1.0 + np.exp(-(r - 0.78) / 0.05))
    elif form == "center_spot":
        shape = 1.0 / (1.0 + np.exp((r - 0.30) / 0.05))
    elif form == "angular":
        shape = np.cos(theta - np.deg2rad(angle_deg))
    else:
        raise ValueError(f"unknown spatial form: {form!r}")
    return magnitude * shape


# --------------------------------------------------------------------------- #
# Synthetic source
# --------------------------------------------------------------------------- #

# Ground-truth scenarios.  ``composite`` and ``borderline`` cases are the ones
# the rule engine is expected to struggle with -- they are the reason the LLM
# adjudication layer exists at all.
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "edge_ring": {
        "terms": [("gate_oxide_thickness", "edge_ring", 0.06)],
        "base_rate": 0.03, "gain": 0.55, "kind": "simple",
    },
    "center_spot": {
        "terms": [("channel_doping", "center_spot", 0.10)],
        "base_rate": 0.03, "gain": 0.50, "kind": "simple",
    },
    "radial_gradient": {
        "terms": [("gate_length", "radial_linear", 0.04)],
        "base_rate": 0.03, "gain": 0.45, "kind": "simple",
    },
    "one_sided": {
        "terms": [("channel_doping", "angular", 0.11)],
        "base_rate": 0.04, "gain": 0.40, "kind": "simple",
    },
    "edge_ring_plus_center": {
        "terms": [("gate_oxide_thickness", "edge_ring", 0.05),
                  ("channel_doping", "center_spot", 0.09)],
        "base_rate": 0.03, "gain": 0.50, "kind": "composite",
    },
    "radial_plus_angular": {
        "terms": [("gate_length", "radial_linear", 0.035),
                  ("channel_doping", "angular", 0.085)],
        "base_rate": 0.04, "gain": 0.45, "kind": "composite",
    },
    "weak_edge_ring": {
        "terms": [("gate_oxide_thickness", "edge_ring", 0.022)],
        "base_rate": 0.05, "gain": 0.30, "kind": "borderline",
    },
    "weak_radial": {
        "terms": [("gate_length", "radial_linear", 0.015)],
        "base_rate": 0.05, "gain": 0.28, "kind": "borderline",
    },
}


def generate_case(case_id: str, scenario: str, n: int = 41,
                  rng: Optional[np.random.Generator] = None) -> WaferCase:
    """Generate one synthetic wafer case with a recorded ground truth."""
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario!r}")
    rng = rng or np.random.default_rng(0)
    spec = SCENARIOS[scenario]
    r, theta, mask = wafer_grid(n)

    angle = float(rng.uniform(0, 360))
    stress = np.zeros_like(r)
    gt_terms = []
    for param, form, mag in spec["terms"]:
        jitter = float(rng.normal(1.0, 0.08))
        mag_j = mag * jitter
        field = spatial_field(form, r, theta, abs(mag_j), angle)
        stress += np.clip(field, 0.0, None) / max(abs(mag_j), 1e-9) * abs(mag_j)
        gt_terms.append({"param": param, "spatial_form": form,
                         "magnitude": round(mag_j, 4),
                         "angle_deg": round(angle, 1) if form == "angular" else None})

    # Convert the variation field into a die failure probability.
    scale = max(float(np.max(stress[mask])), 1e-9)
    p_fail = spec["base_rate"] + spec["gain"] * (stress / scale)
    p_fail = np.clip(p_fail, 0.0, 0.95)

    wmap = np.full((n, n), -1, dtype=int)          # -1 = outside wafer
    draws = rng.random((n, n))
    wmap[mask] = (draws[mask] < p_fail[mask]).astype(int)   # 1 = fail, 0 = pass

    case = WaferCase(
        case_id=case_id,
        wafer_map=Tagged(wmap, Provenance.SYNTHETIC,
                         f"synthetic scenario '{scenario}' (ground truth known)"),
        ground_truth={"scenario": scenario, "kind": spec["kind"], "terms": gt_terms},
    )
    case.log(f"S1: generated synthetic case from scenario '{scenario}'")
    return case


def generate_dataset(n_per_scenario: int = 2, n: int = 41,
                     seed: int = 20260928) -> List[WaferCase]:
    rng = np.random.default_rng(seed)
    cases: List[WaferCase] = []
    for scenario in SCENARIOS:
        for k in range(n_per_scenario):
            cases.append(generate_case(f"{scenario}_{k:02d}", scenario, n, rng))
    return cases


# --------------------------------------------------------------------------- #
# WM-811K source
# --------------------------------------------------------------------------- #

def load_wm811k(path: str, limit: int = 16,
                grid: int = 41) -> List[WaferCase]:
    """Load real wafer maps from a local LSWMD.pkl.

    The file is not redistributed with this repository.  Maps are resampled to
    the working grid and tagged MEASURED; no process ground truth exists, so
    ``ground_truth`` stays ``None`` and every downstream hypothesis is ASSUMED.
    """
    import pickle

    with open(path, "rb") as fh:
        df = pickle.load(fh)

    cases: List[WaferCase] = []
    for idx, row in enumerate(df.itertuples()):
        if len(cases) >= limit:
            break
        raw = np.asarray(getattr(row, "waferMap"))
        if raw.ndim != 2 or min(raw.shape) < 8:
            continue
        wmap = _resample_map(raw, grid)
        label = getattr(row, "failureType", None)
        label = _clean_label(label)
        case = WaferCase(
            case_id=f"wm811k_{idx:05d}",
            wafer_map=Tagged(wmap, Provenance.MEASURED,
                             f"WM-811K map, dataset label={label!r}"),
            ground_truth=None,
        )
        case.log("S1: loaded real WM-811K map (no process ground truth available)")
        if label:
            case.log(f"S1: dataset failure label = {label}")
        cases.append(case)
    return cases


def _resample_map(raw: np.ndarray, grid: int) -> np.ndarray:
    """Nearest-neighbour resample of a WM-811K map onto the working grid.

    WM-811K encodes 0 = outside, 1 = pass, 2 = fail; we convert to the internal
    convention -1 = outside, 0 = pass, 1 = fail.
    """
    h, w = raw.shape
    yi = np.clip((np.arange(grid) * h / grid).astype(int), 0, h - 1)
    xi = np.clip((np.arange(grid) * w / grid).astype(int), 0, w - 1)
    sub = raw[np.ix_(yi, xi)]
    out = np.full((grid, grid), -1, dtype=int)
    out[sub == 1] = 0
    out[sub == 2] = 1
    return out


def _clean_label(label: Any) -> Optional[str]:
    if label is None:
        return None
    arr = np.asarray(label).ravel()
    if arr.size == 0:
        return None
    return str(arr[0])


def load_cases(cfg: Dict[str, Any]) -> List[WaferCase]:
    """Dispatch on ``data.source`` from pipeline.yaml."""
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source", "synthetic")
    grid = int(data_cfg.get("grid", 41))
    if source == "wm811k":
        return load_wm811k(data_cfg.get("wm811k_path", "data/LSWMD.pkl"), grid=grid)
    return generate_dataset(n_per_scenario=int(data_cfg.get("n_per_scenario", 2)),
                            n=grid, seed=int(data_cfg.get("seed", 20260928)))
