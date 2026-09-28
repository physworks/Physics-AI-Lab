"""S1 -- turn a wafer map into numbers, then into a pattern call.

The deliberate design choice here is that S2 never receives a bare label.  It
receives ``SpatialDescriptors`` (continuous, physically interpretable) plus a
``PatternCall`` that is explicit about being composite or borderline.  Those two
flags are what route a case to the LLM adjudication layer.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from ..contracts import (PatternCall, Provenance, SpatialDescriptors, Tagged,
                         WaferCase)
from .data import wafer_grid

N_RADIAL_BINS = 8


def extract_descriptors(wmap: np.ndarray) -> SpatialDescriptors:
    n = wmap.shape[0]
    r, theta, _ = wafer_grid(n)
    inside = wmap >= 0
    fail = wmap == 1

    n_dies = int(inside.sum())
    if n_dies == 0:
        raise ValueError("wafer map contains no valid dies")
    defect_rate = float(fail.sum() / n_dies)

    # Radial profile: defect rate per equal-width radius bin.
    edges = np.linspace(0.0, 1.0, N_RADIAL_BINS + 1)
    profile: List[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = inside & (r >= lo) & (r < hi)
        profile.append(float(fail[sel].sum() / sel.sum()) if sel.sum() > 0 else 0.0)

    inner = inside & (r < 0.30)
    mid = inside & (r >= 0.35) & (r <= 0.65)
    outer = inside & (r > 0.75)
    inner_rate = float(fail[inner].sum() / max(inner.sum(), 1))
    mid_rate = float(fail[mid].sum() / max(mid.sum(), 1))
    outer_rate = float(fail[outer].sum() / max(outer.sum(), 1))
    edge_center_ratio = (outer_rate + 1e-4) / (inner_rate + 1e-4)
    # Referencing the mid annulus keeps an edge ring and a centre spot from
    # cancelling each other out when they occur on the same wafer.
    edge_mid_ratio = (outer_rate + 1e-4) / (mid_rate + 1e-4)
    center_mid_ratio = (inner_rate + 1e-4) / (mid_rate + 1e-4)

    centers = 0.5 * (edges[:-1] + edges[1:])
    slope = float(np.polyfit(centers, profile, 1)[0]) if len(profile) > 1 else 0.0
    curvature = float(np.polyfit(centers, profile, 2)[0]) if len(profile) > 2 else 0.0

    # Angular anisotropy via the first circular moment of failing dies.
    ft = theta[fail & inside]
    if ft.size > 3:
        cx, cy = float(np.mean(np.cos(ft))), float(np.mean(np.sin(ft)))
        anisotropy = float(np.hypot(cx, cy))
        dominant = float(np.rad2deg(np.arctan2(cy, cx)) % 360.0)
    else:
        anisotropy, dominant = 0.0, 0.0

    # Cluster geometry of failing dies.
    fr = r[fail & inside]
    if fr.size > 3:
        centroid_r = float(np.mean(fr))
        compactness = float(max(0.0, 1.0 - np.std(fr) / 0.40))
    else:
        centroid_r, compactness = 0.0, 0.0

    return SpatialDescriptors(
        defect_rate=defect_rate,
        radial_profile=profile,
        edge_center_ratio=float(edge_center_ratio),
        edge_mid_ratio=float(edge_mid_ratio),
        center_mid_ratio=float(center_mid_ratio),
        profile_curvature=curvature,
        radial_slope=slope,
        angular_anisotropy=anisotropy,
        dominant_angle_deg=dominant,
        cluster_centroid_r=centroid_r,
        cluster_compactness=compactness,
        n_dies=n_dies,
    )


# --------------------------------------------------------------------------- #
# Rule-based pattern classification
# --------------------------------------------------------------------------- #

# Each rule maps descriptors to a score in [0, 1].  Scores above ON_THRESHOLD
# fire; two or more firing rules make the call composite.  A score inside
# ON_THRESHOLD +/- MARGIN makes the call borderline.
ON_THRESHOLD = 0.50
MARGIN = 0.12


def _ramp(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def profile_shape(d: SpatialDescriptors) -> Tuple[float, float]:
    """Return (linearity, edge_step) of the radial defect profile.

    An edge ring and a radial gradient both raise the edge/centre ratio, so that
    ratio alone cannot separate them.  The shape of the profile can: a ring is a
    step confined to the outer bins (poor linear fit, large step), a gradient is
    a steady rise across all bins (good linear fit, small step).
    """
    prof = np.asarray(d.radial_profile, dtype=float)
    if prof.size < 4:
        return 0.0, 0.0
    centers = np.linspace(0.0, 1.0, prof.size)
    fit = np.polyval(np.polyfit(centers, prof, 1), centers)
    ss_res = float(((prof - fit) ** 2).sum())
    ss_tot = float(((prof - prof.mean()) ** 2).sum())
    linearity = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    k = max(1, prof.size // 4)
    inner = float(prof[:-2 * k].mean()) if prof.size > 2 * k else float(prof[:k].mean())
    outer = float(prof[-k:].mean())
    scale = float(prof.max()) + 1e-6
    edge_step = (outer - inner) / scale
    return float(np.clip(linearity, 0.0, 1.0)), float(edge_step)


def score_patterns(d: SpatialDescriptors) -> Dict[str, float]:
    linearity, edge_step = profile_shape(d)
    edge_heavy = _ramp(d.edge_mid_ratio, 1.25, 2.6)
    center_heavy = _ramp(d.center_mid_ratio, 1.25, 2.6)

    # A ring is a step confined to the outer bins: it scores on edge weight and
    # on departure from a straight profile.  A gradient is the opposite -- slope
    # with a good linear fit.
    ring = edge_heavy * (1.0 - 0.85 * _ramp(linearity, 0.72, 0.96))
    gradient = _ramp(abs(d.radial_slope), 0.05, 0.25) \
        * (0.30 + 0.70 * _ramp(linearity, 0.80, 0.95))
    spot = center_heavy * (0.5 + 0.5 * _ramp(0.55 - d.cluster_centroid_r, 0.02, 0.25))

    return {
        "edge_ring": float(np.clip(ring, 0.0, 1.0)),
        "center_spot": float(np.clip(spot, 0.0, 1.0)),
        "radial_gradient": float(np.clip(gradient, 0.0, 1.0)),
        "one_sided": float(_ramp(d.angular_anisotropy, 0.12, 0.42)),
    }


def classify(d: SpatialDescriptors) -> PatternCall:
    scores = score_patterns(d)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = ranked[0]

    fired = [k for k, v in scores.items() if v >= ON_THRESHOLD]
    is_composite = len(fired) >= 2
    borderline_notes: List[str] = []
    for k, v in scores.items():
        if abs(v - ON_THRESHOLD) <= MARGIN:
            borderline_notes.append(
                f"{k} score {v:.2f} sits within +/-{MARGIN} of the {ON_THRESHOLD} threshold")
    is_borderline = len(borderline_notes) > 0

    label = top if top_score >= ON_THRESHOLD else "none"
    if is_composite:
        label = "+".join(sorted(fired))
    return PatternCall(label=label, scores=scores, is_composite=is_composite,
                       is_borderline=is_borderline, margin_notes=borderline_notes)


def run(case: WaferCase) -> WaferCase:
    """S1 stage entry point."""
    wmap = case.wafer_map.value
    d = extract_descriptors(wmap)
    p = classify(d)

    # Descriptors inherit the provenance of the map they were computed from.
    case.descriptors = Tagged(d, case.wafer_map.provenance,
                              "computed directly from the wafer map")
    case.pattern = Tagged(p, case.wafer_map.provenance,
                          "deterministic rule-based classification")
    case.log(f"S1: {d.summary()}")
    case.log(f"S1: pattern={p.label} composite={p.is_composite} "
             f"borderline={p.is_borderline}")
    return case
