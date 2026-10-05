"""The fixed stage order, and where the one replan loop is allowed to run.

    nominal parameters
        -> screening Fisher analysis        (evidence, computed)
        -> objective decision               (agent or rule: THE judgement)
        -> constrained greedy selection     (computed, identical for all arms)
        -> feasibility gate                 (hard rules, nobody may relax)
        -> [replan, at most `max_replans`]
        -> Monte Carlo evaluation           (computed)

The loop is bounded and its exit conditions are written down: a design either
satisfies the constraints or the stage escalates to a human with the list of
violations.  It never proceeds on a design that failed the gate, and it never
rewrites the constraint that was violated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

from .agent import AgentDecision, decide
from .constraints import CandidatePool, DesignConstraints
from .device import DeviceParams
from .fisher import information
from .noise import NoiseModel
from .objective import Objective, evidence_from, full_objective
from .select import Design, greedy_design


@dataclass
class StageOutcome:
    design: Design | None
    decision: AgentDecision | None
    evidence: Dict
    replans: int = 0
    escalated: bool = False
    log: List[str] = field(default_factory=list)

    def provenance(self) -> Dict:
        return {"escalated": self.escalated, "replans": self.replans,
                "agent": self.decision.provenance() if self.decision else None,
                "design": self.design.summary() if self.design else None,
                "log": list(self.log)}


def screen(p: DeviceParams, pool: CandidatePool, noise: NoiseModel,
           provenance: Dict[str, Dict] | None = None) -> Dict:
    """Evidence for the objective decision: what a full sweep would reveal.

    Deliberately computed on the *whole candidate pool*, i.e. the information
    available if everything allowed were measured.  A parameter that is
    undeterminable even then cannot be rescued by choosing a subset.
    """
    idx = pool.allowed_index()
    report = information(p, pool.vgs[idx], pool.vds[idx], noise)
    return evidence_from(report, p.to_dict(), provenance)


def run_stage(p: DeviceParams, pool: CandidatePool,
              constraints: DesignConstraints, noise: NoiseModel,
              purpose: str, arm: str = "agent",
              max_replans: int = 1,
              transport: Callable | None = None,
              api_key: str | None = None) -> StageOutcome:
    """One pass of design: evidence -> objective -> points -> gate."""
    log: List[str] = []
    evidence = screen(p, pool, noise)
    log.append(f"screened {len(pool.allowed_index())} feasible candidates "
               f"of {len(pool)}")

    decision: AgentDecision | None = None
    if arm == "agent":
        decision = decide(evidence, constraints.describe(), p.to_dict(),
                          purpose, transport=transport, api_key=api_key)
        objective = decision.objective
        log.append("objective from "
                   f"{'model' if decision.used_llm else 'rule fallback'}: "
                   f"{objective.criterion} over {objective.include_params}")
    elif arm == "textbook":
        objective = full_objective(p.to_dict(), "D")
        log.append("objective fixed: D-optimal over all parameters")
    else:
        raise ValueError(f"unknown arm {arm!r}")

    design = greedy_design(p, pool, constraints, noise, objective, arm=arm)
    replans = 0

    while design.violations and replans < max_replans:
        replans += 1
        log.append(f"replan {replans}: {design.violations}")
        relaxed = Objective(include_params=objective.include_params,
                            criterion="A", c_target=None,
                            fixed_params=objective.fixed_params,
                            rationale="replan after an infeasible design; "
                                      "A is less prone to degenerate points",
                            source=objective.source)
        design = greedy_design(p, pool, constraints, noise, relaxed, arm=arm)
        objective = relaxed

    escalated = bool(design.violations)
    if escalated:
        log.append("escalating to a human: constraints still unsatisfied")

    return StageOutcome(design=None if escalated else design,
                        decision=decision, evidence=evidence,
                        replans=replans, escalated=escalated, log=log)
