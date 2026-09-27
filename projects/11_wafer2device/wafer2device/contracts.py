"""Data contracts shared by every stage of the pipeline.

Design rule: a single ``WaferCase`` object flows through the DAG and each stage
fills in its own fields.  Every value that matters carries a ``Provenance`` tag
so that the final report can separate what was *measured* from what was
*assumed*, *predicted* or *optimized*.  This separation is the backbone of the
project's honesty story -- it is enforced by the type system, not by prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
import time


class Provenance(str, Enum):
    """Where a value came from.  Never guess -- tag it."""

    MEASURED = "measured"      # real data (wafer map, descriptors computed from it)
    SYNTHETIC = "synthetic"    # generated data with a known ground truth
    ASSUMED = "assumed"        # physical hypothesis, human-approved
    PREDICTED = "predicted"    # model output
    OPTIMIZED = "optimized"    # optimizer output


@dataclass
class Tagged:
    """A value plus its provenance and a short human-readable note."""

    value: Any
    provenance: Provenance
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"value": _jsonable(self.value),
                "provenance": self.provenance.value,
                "note": self.note}


# --------------------------------------------------------------------------- #
# S1: wafer map -> spatial descriptors
# --------------------------------------------------------------------------- #

@dataclass
class SpatialDescriptors:
    """Quantitative descriptors extracted from a wafer map.

    The point of this stage is that downstream reasoning (rule *or* LLM) is
    grounded in numbers, not in a bare class label.
    """

    defect_rate: float                      # overall fraction of failing dies
    radial_profile: List[float]             # defect rate per normalised radius bin
    edge_center_ratio: float                # outer-ring rate / inner-disc rate
    radial_slope: float                     # linear fit slope of radial_profile
    angular_anisotropy: float               # 0 = isotropic, 1 = fully one-sided
    dominant_angle_deg: float               # direction of the anisotropy
    cluster_centroid_r: float               # centroid radius of failing dies (0..1)
    cluster_compactness: float              # 0 = diffuse, 1 = tight cluster
    edge_mid_ratio: float = 1.0             # outer-ring rate / mid-annulus rate
    center_mid_ratio: float = 1.0           # inner-disc rate / mid-annulus rate
    profile_curvature: float = 0.0          # quadratic coefficient of the profile
    n_dies: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (f"defect_rate={self.defect_rate:.3f}, "
                f"edge/mid={self.edge_mid_ratio:.2f}, "
                f"center/mid={self.center_mid_ratio:.2f}, "
                f"radial_slope={self.radial_slope:+.3f}, "
                f"anisotropy={self.angular_anisotropy:.2f}, "
                f"centroid_r={self.cluster_centroid_r:.2f}, "
                f"compactness={self.cluster_compactness:.2f}")


@dataclass
class PatternCall:
    """Rule-based pattern classification with the evidence behind it."""

    label: str
    scores: Dict[str, float]
    is_composite: bool = False           # two or more patterns fire together
    is_borderline: bool = False          # score sits inside the decision margin
    margin_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# S2: descriptors -> process variation hypothesis
# --------------------------------------------------------------------------- #

@dataclass
class VariationTerm:
    """One process-variation term of a hypothesis.

    ``param`` and ``spatial_form`` must come from the whitelist in
    ``configs/whitelist.yaml``.  Free-text parameters are rejected before they
    ever reach the device model.
    """

    param: str                # e.g. "gate_oxide_thickness"
    spatial_form: str         # e.g. "radial_linear", "edge_ring", "uniform"
    magnitude: float          # fractional deviation, e.g. 0.05 == +5%
    confidence: float = 0.5   # self-reported; advisory only, never a gate
    angle_deg: Optional[float] = None   # only meaningful for spatial_form "angular"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def key(self) -> Tuple[str, str, float]:
        """Identity used to detect a repeated hypothesis across replans."""
        return (self.param, self.spatial_form, round(self.magnitude, 4))


@dataclass
class Hypothesis:
    """A complete, whitelist-validated explanation of the observed pattern."""

    terms: List[VariationTerm]
    rationale: str = ""
    source: str = "rule"            # "rule" | "llm" | "llm_refine" | "llm_dispute"
    adjudication: Optional[str] = None   # agree | refine | dispute (LLM verdict)
    replan_round: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"terms": [t.to_dict() for t in self.terms],
                "rationale": self.rationale,
                "source": self.source,
                "adjudication": self.adjudication,
                "replan_round": self.replan_round}

    def signature(self) -> Tuple:
        return tuple(sorted(t.key() for t in self.terms))


# --------------------------------------------------------------------------- #
# S3/S4: device response and compensation
# --------------------------------------------------------------------------- #

@dataclass
class DeviceResponse:
    """Per-die device metrics predicted from a variation field."""

    model_name: str
    vth: List[float]
    ion: List[float]
    ioff: List[float]
    out_of_spec_rate: float
    stats: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"model_name": self.model_name,
                "out_of_spec_rate": self.out_of_spec_rate,
                "stats": self.stats,
                "n_dies": len(self.vth)}


@dataclass
class CompensationResult:
    """Robust design point found by the optimizer."""

    knobs: Dict[str, float]
    baseline_out_of_spec: float
    optimized_out_of_spec: float
    improvement: float
    n_evaluations: int
    history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["history"] = [float(x) for x in self.history[:50]]
        return d


# --------------------------------------------------------------------------- #
# Physics checks and replanning
# --------------------------------------------------------------------------- #

@dataclass
class CheckResult:
    """Outcome of one physics check.

    ``direction_hint`` is computed by the check itself -- it is the physics
    telling the hypothesis generator which way to move, not the engineer
    handing the model an answer.
    """

    name: str
    passed: bool
    observed: float
    threshold: float
    deviation_ratio: float          # 0.0 when passing; >0 = how far out of bounds
    direction_hint: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CheckReport:
    results: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def violation_score(self) -> float:
        """Scalar severity used by the stall-detection stop rule."""
        return sum(r.deviation_ratio for r in self.results if not r.passed)

    def failures(self) -> List[CheckResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed,
                "violation_score": self.violation_score,
                "results": [r.to_dict() for r in self.results]}


@dataclass
class ReplanAttempt:
    """One turn of the replanning loop, kept for the portfolio's audit trail."""

    round_index: int
    hypothesis: Hypothesis
    check_report: CheckReport
    violation_score: float
    stop_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"round_index": self.round_index,
                "hypothesis": self.hypothesis.to_dict(),
                "check_report": self.check_report.to_dict(),
                "violation_score": self.violation_score,
                "stop_reason": self.stop_reason}


@dataclass
class ApprovalRecord:
    """Human decision at the physics gate."""

    decision: str                   # approved | modified | rejected | escalated
    approver: str = "human"
    comment: str = ""
    modified_terms: Optional[List[Dict[str, Any]]] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# The object that flows through the DAG
# --------------------------------------------------------------------------- #

@dataclass
class WaferCase:
    case_id: str
    wafer_map: Tagged                                   # 2-D array, measured|synthetic
    ground_truth: Optional[Dict[str, Any]] = None       # only for synthetic cases

    descriptors: Optional[Tagged] = None                # SpatialDescriptors
    pattern: Optional[Tagged] = None                    # PatternCall
    rule_hypothesis: Optional[Tagged] = None            # Hypothesis (rule path)
    hypothesis: Optional[Tagged] = None                 # Hypothesis (final, assumed)
    device_response: Optional[Tagged] = None            # DeviceResponse
    compensation: Optional[Tagged] = None               # CompensationResult

    check_report: Optional[CheckReport] = None
    replan_log: List[ReplanAttempt] = field(default_factory=list)
    approval: Optional[ApprovalRecord] = None
    escalated: bool = False
    notes: List[str] = field(default_factory=list)

    # -- helpers ---------------------------------------------------------- #

    def log(self, msg: str) -> None:
        self.notes.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        def t(x):
            return x.to_dict() if isinstance(x, Tagged) else None

        return {
            "case_id": self.case_id,
            "wafer_map_provenance": self.wafer_map.provenance.value,
            "ground_truth": _jsonable(self.ground_truth),
            "descriptors": t(self.descriptors),
            "pattern": t(self.pattern),
            "rule_hypothesis": t(self.rule_hypothesis),
            "hypothesis": t(self.hypothesis),
            "device_response": t(self.device_response),
            "compensation": t(self.compensation),
            "check_report": self.check_report.to_dict() if self.check_report else None,
            "replan_log": [a.to_dict() for a in self.replan_log],
            "approval": self.approval.to_dict() if self.approval else None,
            "escalated": self.escalated,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion so that any Tagged value can be serialised."""
    import numpy as np

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Provenance):
        return obj.value
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)
