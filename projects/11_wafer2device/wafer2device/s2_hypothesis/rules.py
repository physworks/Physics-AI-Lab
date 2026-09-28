"""S2 (rule path) -- descriptors to a process-variation hypothesis.

This path is fully deterministic and always available: no API key, no network,
no cache.  It is the safety net *and* the control arm of the rule-vs-LLM
comparison.  Making it genuinely good is the point -- an LLM that only beats a
strawman has not been shown to be worth its cost.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from ..contracts import (Hypothesis, PatternCall, Provenance, SpatialDescriptors,
                         Tagged, VariationTerm, WaferCase)
from .schema import Whitelist

# Known pattern -> physical cause mappings.  These are the textbook
# correspondences that make the rule engine a fair baseline.
PATTERN_TO_TERM: Dict[str, Dict[str, str]] = {
    "edge_ring": {"param": "gate_oxide_thickness", "spatial_form": "edge_ring"},
    "center_spot": {"param": "channel_doping", "spatial_form": "center_spot"},
    "radial_gradient": {"param": "gate_length", "spatial_form": "radial_linear"},
    "one_sided": {"param": "channel_doping", "spatial_form": "angular"},
}

# Defect-rate contrast is converted into a magnitude through a fixed gain.
# The gain is a modelling assumption, recorded here rather than hidden in code.
CONTRAST_GAIN = 0.09


# Smallest magnitude a fired rule may propose.  A rule that names a pattern but
# then asks for zero deviation is self-contradictory: it would claim a cause and
# simultaneously claim nothing varies.  Such a hypothesis must be a deliberate
# null, never an artefact of the contrast formula bottoming out.
MIN_FIRED_MAGNITUDE = 0.015


def _magnitude_for(pattern: str, d: SpatialDescriptors, wl: Whitelist) -> float:
    # The contrast measure must use the same reference annulus as the pattern
    # score in S1.  Scoring against the mid annulus while sizing the magnitude
    # against the opposite side of the wafer makes the two disagree exactly when
    # an edge ring and a centre spot occur together -- the composite case.
    if pattern == "edge_ring":
        contrast = np.log10(max(d.edge_mid_ratio, 1e-3))
    elif pattern == "center_spot":
        contrast = np.log10(max(d.center_mid_ratio, 1e-3))
    elif pattern == "radial_gradient":
        contrast = abs(d.radial_slope) * 6.0
    elif pattern == "one_sided":
        contrast = d.angular_anisotropy * 2.0
    else:
        contrast = 0.0

    mag = CONTRAST_GAIN * float(np.clip(contrast, 0.0, 3.0))
    mag = max(mag, MIN_FIRED_MAGNITUDE)
    spec = wl.parameters[PATTERN_TO_TERM[pattern]["param"]]
    return float(np.clip(mag, spec["min"], spec["max"]))


def propose(d: SpatialDescriptors, p: PatternCall, wl: Whitelist) -> Hypothesis:
    """Deterministic hypothesis from descriptors and the pattern call.

    Known limitation, kept on purpose: for a composite call the rule engine
    takes only the single highest-scoring pattern.  It has no mechanism for
    apportioning a wafer signature between two simultaneous causes.
    """
    ranked = sorted(p.scores.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = ranked[0]

    if top not in PATTERN_TO_TERM or top_score < 0.20:
        return Hypothesis(
            terms=[VariationTerm("channel_doping", "uniform", 0.0, confidence=0.1)],
            rationale="No pattern rule fired with meaningful confidence; "
                      "defaulting to a null (uniform, zero) hypothesis.",
            source="rule")

    mapping = PATTERN_TO_TERM[top]
    term = VariationTerm(param=mapping["param"],
                         spatial_form=mapping["spatial_form"],
                         magnitude=_magnitude_for(top, d, wl),
                         confidence=float(np.clip(top_score, 0.0, 1.0)),
                         angle_deg=(d.dominant_angle_deg
                                    if mapping["spatial_form"] == "angular" else None))
    rationale = (f"Highest-scoring rule is '{top}' (score {top_score:.2f}); "
                 f"mapped to {term.param}/{term.spatial_form} with magnitude "
                 f"{term.magnitude:+.3f} via a fixed contrast gain.")
    if p.is_composite:
        others = [k for k, v in p.scores.items() if v >= 0.5 and k != top]
        rationale += (f" NOTE: {', '.join(others)} also fired but the rule engine "
                      f"cannot decompose composite signatures.")
    return Hypothesis(terms=[term], rationale=rationale, source="rule")


def revise(previous: Hypothesis, failures: List[Dict[str, object]],
           wl: Whitelist, round_index: int) -> Hypothesis:
    """Rule-only replanning: nudge magnitudes along the check's direction hint.

    This exists so that the rule arm of the comparison also gets to replan --
    otherwise the LLM would win the replanning metric by default.  What it
    cannot do is change *which* parameter is blamed, which is exactly the
    structural difference measured by the experiment.
    """
    step = 0.6 ** round_index
    new_terms: List[VariationTerm] = []
    for t in previous.terms:
        delta = 0.0
        for f in failures:
            hint = str(f.get("direction_hint", ""))
            if t.param in hint and "decrease" in hint:
                delta -= CONTRAST_GAIN * step
            elif t.param in hint and "increase" in hint:
                delta += CONTRAST_GAIN * step
        spec = wl.parameters[t.param]
        mag = float(np.clip(t.magnitude + delta, spec["min"], spec["max"]))
        new_terms.append(VariationTerm(t.param, t.spatial_form, mag,
                                       confidence=max(0.1, t.confidence - 0.1),
                                       angle_deg=t.angle_deg))
    return Hypothesis(terms=new_terms,
                      rationale=f"Rule-based replan round {round_index}: magnitudes "
                                f"nudged along physics-check direction hints.",
                      source="rule", replan_round=round_index)
