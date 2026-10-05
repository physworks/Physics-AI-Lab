"""Optimal measurement design for compact model parameter extraction.

Stage 1 of ``13_measurement-to-circuit``.  The question: given a compact
model, an instrument with a noise floor, and limits on what may be measured,
which bias points should be measured so the extracted parameters are worth
trusting?
"""

from .agent import AgentDecision, decide
from .constraints import (CandidatePool, DesignConstraints, build_pool,
                          default_required_regions, violations)
from .device import DeviceParams, PARAM_NAMES, extract_subset
from .evaluate import ArmResult, equivalent_points, holdout_grid, run_arm
from .fisher import FisherResult, evaluate_design, information, score
from .noise import NoiseModel
from .objective import (Objective, ObjectiveRejected, full_objective,
                        rule_objective, validate)
from .pipeline import StageOutcome, run_stage, screen
from .select import (Design, greedy_design, heuristic_design, uniform_design)

__all__ = [
    "AgentDecision", "ArmResult", "CandidatePool", "Design",
    "DesignConstraints", "DeviceParams", "FisherResult", "NoiseModel",
    "Objective", "ObjectiveRejected", "PARAM_NAMES", "StageOutcome",
    "build_pool", "decide", "default_required_regions", "equivalent_points",
    "evaluate_design", "extract_subset", "full_objective", "greedy_design",
    "heuristic_design", "holdout_grid", "information", "rule_objective",
    "run_arm", "run_stage", "score", "screen", "uniform_design", "validate",
    "violations",
]
