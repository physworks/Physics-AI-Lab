"""What the design is trying to learn -- the one decision the agent owns.

An objective is two choices:

  * **which parameters** the design should try to pin down, and
  * **which criterion** reduces their joint uncertainty to one number.

Neither is a calculation.  Both depend on what the model is *for*, and on
evidence about which parameters the data can determine at all.  Project 12
found that ``theta`` has 1/87 of ``vth0``'s sensitivity: no measurement plan
recovers it, and leaving it in a D-optimal objective makes the determinant
chase a direction that carries no information, at the cost of the parameters
that do.  Dropping it is a modelling judgement, and a wrong one if the model
will later be used in a regime where ``theta`` matters.

A parameter dropped from the objective is **held fixed during extraction**
too.  Deciding a parameter is undeterminable and then letting the optimiser
vary it anyway would make the decision cosmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .device import PARAM_NAMES
from .fisher import CRITERIA


@dataclass(frozen=True)
class Objective:
    include_params: List[str]
    criterion: str = "D"
    c_target: str | None = None
    fixed_params: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    source: str = "rule"               # rule | llm | config

    def describe(self) -> Dict:
        return {"include_params": list(self.include_params),
                "criterion": self.criterion, "c_target": self.c_target,
                "fixed_params": dict(self.fixed_params),
                "source": self.source, "rationale": self.rationale}


class ObjectiveRejected(ValueError):
    """The proposed objective was outside what the configuration permits."""


def validate(proposal: Dict, allowed_params: Sequence[str],
             nominal: Dict[str, float],
             min_params: int = 2) -> Objective:
    """Whitelist enforcement.

    Anything a model proposes passes through here before it can influence a
    design.  Unknown keys, unknown parameters, an unknown criterion or an
    empty objective are refused rather than repaired, so a malformed
    suggestion falls back to the rule arm instead of silently becoming
    something else.
    """
    if not isinstance(proposal, dict):
        raise ObjectiveRejected("proposal is not an object")

    extra = set(proposal) - {"include_params", "criterion", "c_target",
                             "rationale"}
    if extra:
        raise ObjectiveRejected(f"unexpected key(s): {sorted(extra)}")

    include = proposal.get("include_params")
    if not isinstance(include, list) or not include:
        raise ObjectiveRejected("include_params must be a non-empty list")
    include = [str(x) for x in include]

    unknown = [x for x in include if x not in allowed_params]
    if unknown:
        raise ObjectiveRejected(f"parameter(s) not permitted: {unknown}")
    if len(set(include)) != len(include):
        raise ObjectiveRejected("include_params contains duplicates")
    if len(include) < min_params:
        raise ObjectiveRejected(
            f"at least {min_params} parameters must be estimated")

    criterion = str(proposal.get("criterion", "D"))
    if criterion not in CRITERIA:
        raise ObjectiveRejected(f"criterion must be one of {CRITERIA}")

    c_target = proposal.get("c_target")
    if criterion == "c":
        if c_target not in include:
            raise ObjectiveRejected(
                "criterion 'c' needs a c_target drawn from include_params")
    else:
        c_target = None

    fixed = {n: float(nominal[n]) for n in PARAM_NAMES if n not in include}
    return Objective(include_params=include, criterion=criterion,
                     c_target=c_target, fixed_params=fixed,
                     rationale=str(proposal.get("rationale", ""))[:500],
                     source="llm")


def full_objective(nominal: Dict[str, float], criterion: str = "D") -> Objective:
    """Textbook default: every parameter, equally weighted."""
    return Objective(include_params=list(PARAM_NAMES), criterion=criterion,
                     fixed_params={}, source="config",
                     rationale="all parameters, no identifiability screening")


def rule_objective_provenance(evidence: Dict, nominal: Dict[str, float],
                              sensitivity_ratio: float = 0.02,
                              trust_tolerance: float = 0.05) -> Objective:
    """The stronger rule: drop a parameter only if its nominal is trustworthy.

    Fixing an undeterminable parameter removes a nuisance direction, which
    helps -- but only if the value it is fixed *at* is right.  Fixed at a
    stale default it injects a bias no measurement plan can undo.  So this
    rule reads the provenance block as well as the sensitivities.

    It is a fair baseline rather than a strawman: on the two cases its author
    had in mind it gets the same answer the agent does.  What it cannot do is
    respond to a purpose nobody encoded -- see ``outputs/report.md``.
    """
    sens = {k: float(v) for k, v in evidence.get("sensitivity", {}).items()}
    prov = evidence.get("nominal_provenance", {}) or {}
    if not sens:
        return full_objective(nominal)

    smax = max(sens.values()) or 1.0
    drop = []
    for n in PARAM_NAMES:
        weak = sens.get(n, 0.0) / smax < sensitivity_ratio
        trust = float(prov.get(n, {}).get("rel_uncertainty", 1.0))
        if weak and trust <= trust_tolerance:
            drop.append(n)

    keep = [n for n in PARAM_NAMES if n not in drop]
    if len(keep) < 2:
        keep, drop = list(PARAM_NAMES), []

    why = ("dropped " + (", ".join(drop) if drop else "nothing")
           + " (low sensitivity and a trusted nominal); kept the rest")
    return Objective(include_params=keep, criterion="D",
                     fixed_params={n: float(nominal[n]) for n in drop},
                     rationale=why, source="rule+provenance")


def rule_objective(evidence: Dict, nominal: Dict[str, float],
                   sensitivity_ratio: float = 0.02,
                   sigma_reject: float = 0.05) -> Objective:
    """Deterministic fallback, and the baseline the agent has to beat.

    Two screens, both defensible and both blunt:

      * drop a parameter whose sensitivity is below ``sensitivity_ratio`` of
        the largest -- the data barely responds to it;
      * drop a parameter whose nominal relative 1-sigma already exceeds
        ``sigma_reject`` -- it is not going to be determined.

    Then pick ``A`` when the surviving uncertainties are badly spread, since
    the determinant would be dominated by the worst one, and ``D`` otherwise.
    This is a reasonable rule.  What it cannot do is notice that a parameter
    it is about to drop is the one the model will be judged on.
    """
    sens = {k: float(v) for k, v in evidence.get("sensitivity", {}).items()}
    sig = {k: float(v) for k, v in evidence.get("rel_sigma", {}).items()}
    if not sens:
        return full_objective(nominal)

    smax = max(sens.values()) or 1.0
    keep = [n for n in PARAM_NAMES
            if sens.get(n, 0.0) / smax >= sensitivity_ratio
            and sig.get(n, 0.0) <= sigma_reject]
    dropped = [n for n in PARAM_NAMES if n not in keep]

    if len(keep) < 2:
        keep, dropped = list(PARAM_NAMES), []

    kept_sig = [sig[n] for n in keep if n in sig]
    spread = (max(kept_sig) / min(kept_sig)) if len(kept_sig) >= 2 else 1.0
    criterion = "A" if spread > 20.0 else "D"

    why = (f"kept {len(keep)} of {len(PARAM_NAMES)} parameters "
           f"(dropped {dropped or 'none'}); sigma spread {spread:.0f}x "
           f"-> criterion {criterion}")
    return Objective(include_params=keep, criterion=criterion,
                     fixed_params={n: float(nominal[n]) for n in dropped},
                     rationale=why, source="rule")


def evidence_from(report, nominal: Dict[str, float],
                  provenance: Dict[str, Dict] | None = None) -> Dict:
    """Package a Fisher result as the evidence an objective decision needs.

    ``provenance`` says where each nominal value came from and how far it can
    be trusted.  It is not derivable from the data being designed for, and it
    decides whether fixing a parameter is a simplification or a bias.
    """
    corr = []
    if report.correlation is not None:
        c = report.correlation
        for i in range(c.shape[0]):
            for j in range(i + 1, c.shape[1]):
                if abs(c[i, j]) >= 0.90:
                    corr.append([report.names[i], report.names[j],
                                 round(float(c[i, j]), 4)])
    sens = {}
    for n in report.names:
        s = report.rel_sigma.get(n, np.nan)
        sens[n] = float(1.0 / s) if s and np.isfinite(s) and s > 0 else 0.0

    return {"parameters": list(PARAM_NAMES),
            "nominal": {k: float(v) for k, v in nominal.items()},
            "nominal_provenance": dict(provenance or {}),
            "rel_sigma": {k: float(v) for k, v in report.rel_sigma.items()},
            "sensitivity": sens,
            "strong_correlations": corr,
            "condition_number": float(report.condition_number)}
