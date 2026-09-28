"""Whitelist enforcement for S2.

Any hypothesis -- rule-generated or LLM-generated -- must pass through
``validate_hypothesis`` before it can reach the device model.  A parameter that
is not in ``configs/whitelist.yaml`` is not "interpreted" or "mapped"; it is
rejected.  This is the boundary that keeps a language model from silently
widening the physics search space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..contracts import Hypothesis, VariationTerm


class WhitelistError(ValueError):
    """Raised when a proposed hypothesis leaves the allowed vocabulary."""


@dataclass
class Whitelist:
    parameters: Dict[str, Dict[str, Any]]
    spatial_forms: Dict[str, Dict[str, Any]]
    limits: Dict[str, Any]

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Whitelist":
        return cls(parameters=cfg["parameters"],
                   spatial_forms=cfg["spatial_forms"],
                   limits=cfg.get("limits", {}))

    def param_names(self) -> List[str]:
        return list(self.parameters)

    def form_names(self) -> List[str]:
        return list(self.spatial_forms)

    def describe(self) -> str:
        """Compact description handed to the LLM as its allowed vocabulary."""
        lines = ["parameters (fractional deviation, [min, max]):"]
        for name, spec in self.parameters.items():
            lines.append(f"  - {name}: [{spec['min']}, {spec['max']}] -- {spec['description']}")
        lines.append("spatial_forms:")
        for name, spec in self.spatial_forms.items():
            lines.append(f"  - {name}: {spec['description']}")
        lines.append(f"limits: max_terms={self.limits.get('max_terms', 3)}")
        return "\n".join(lines)


def validate_term(term: VariationTerm, wl: Whitelist) -> Tuple[bool, str]:
    if term.param not in wl.parameters:
        return False, f"parameter {term.param!r} is not whitelisted"
    if term.spatial_form not in wl.spatial_forms:
        return False, f"spatial_form {term.spatial_form!r} is not whitelisted"
    spec = wl.parameters[term.param]
    if not (spec["min"] <= term.magnitude <= spec["max"]):
        return False, (f"magnitude {term.magnitude:+.4f} for {term.param} is outside "
                       f"[{spec['min']}, {spec['max']}]")
    hard = wl.limits.get("max_abs_magnitude", 1.0)
    if abs(term.magnitude) > hard:
        return False, f"magnitude {term.magnitude:+.4f} exceeds the hard limit {hard}"
    if not (0.0 <= term.confidence <= 1.0):
        return False, f"confidence {term.confidence} is outside [0, 1]"
    return True, ""


def validate_hypothesis(h: Hypothesis, wl: Whitelist) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    max_terms = int(wl.limits.get("max_terms", 3))
    if not h.terms:
        errors.append("hypothesis contains no terms")
    if len(h.terms) > max_terms:
        errors.append(f"hypothesis has {len(h.terms)} terms, limit is {max_terms}")
    for t in h.terms:
        ok, msg = validate_term(t, wl)
        if not ok:
            errors.append(msg)
    return (len(errors) == 0), errors


def coerce_terms(raw_terms: List[Dict[str, Any]], wl: Whitelist) -> List[VariationTerm]:
    """Build VariationTerm objects from raw JSON, dropping malformed entries.

    Dropping (rather than repairing) is intentional: a hypothesis that needs
    repairing is one the engineer should see fail, not one the code should
    quietly fix.
    """
    terms: List[VariationTerm] = []
    for raw in raw_terms:
        try:
            term = VariationTerm(
                param=str(raw["param"]),
                spatial_form=str(raw["spatial_form"]),
                magnitude=float(raw["magnitude"]),
                confidence=float(raw.get("confidence", 0.5)),
                angle_deg=(float(raw["angle_deg"])
                           if raw.get("angle_deg") is not None else None),
            )
        except (KeyError, TypeError, ValueError):
            continue
        ok, _ = validate_term(term, wl)
        if ok:
            terms.append(term)
    return terms
