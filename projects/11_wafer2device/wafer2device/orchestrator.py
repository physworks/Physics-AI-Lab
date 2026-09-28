"""Hybrid orchestration: a fixed DAG with a bounded replanning loop.

Stage order never changes -- S1 -> S2 -> checks -> gate -> S3 -> S4 -> S5 -- so
every run is reproducible and every failure is attributable to a known stage.
The autonomy lives in exactly one place: when the physics checks fail, S2 is
asked to revise its hypothesis using structured failure feedback.

Stop rules (all three, from the design):
  1. hard cap             max_rounds
  2. stall detection      violation score improves by less than min_improvement
  3. escalation on stop   a loop that never passes is escalated to the engineer,
                          which is a normal outcome, not a crash

An LLM is never given control of the stage order and can never bypass the gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .contracts import (CheckReport, Hypothesis, Provenance, ReplanAttempt,
                        Tagged, WaferCase)
from .gate import request_approval
from .physics_checks.checks import build_feedback, run_all
from .s1_pattern import descriptors as s1
from .s2_hypothesis import llm as s2_llm
from .s2_hypothesis import rules as s2_rules
from .s2_hypothesis.schema import Whitelist
from .s3_device.models import DeviceModel, build_model, variation_fields
from .s4_compensate.optimize import optimize


class Pipeline:
    def __init__(self, pipeline_cfg: Dict[str, Any], checks_cfg: Dict[str, Any],
                 whitelist_cfg: Dict[str, Any]):
        self.cfg = pipeline_cfg
        self.checks_cfg = checks_cfg
        self.wl = Whitelist.from_config(whitelist_cfg)

        dm = pipeline_cfg["device_model"]
        self.model: DeviceModel = build_model(dm.get("primary", "analytic"), dm)
        try:
            self.secondary = build_model(dm.get("secondary"), dm)
        except FileNotFoundError as exc:
            self.secondary = None
            self._secondary_note = str(exc)
        else:
            self._secondary_note = ""

        self.log_dir = Path(pipeline_cfg["report"]["log_dir"])
        self.out_dir = Path(pipeline_cfg["report"]["output_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -- S2 with the replanning loop -------------------------------------- #

    def _propose(self, case: WaferCase, feedback: Optional[Dict[str, Any]],
                 round_index: int, previous: List[Hypothesis]) -> Hypothesis:
        d = case.descriptors.value
        p = case.pattern.value
        mode = self.cfg["hypothesis"]["mode"]

        if feedback is None:
            rule_h = s2_rules.propose(d, p, self.wl)
            case.rule_hypothesis = Tagged(rule_h, Provenance.ASSUMED,
                                          "deterministic rule engine")
        else:
            rule_h = s2_rules.revise(previous[-1], feedback["failures"], self.wl,
                                     round_index)

        if mode == "rule":
            rule_h.replan_round = round_index
            return rule_h

        llm_cfg = dict(self.cfg["hypothesis"]["llm"])
        if llm_cfg.get("escalate_only") and not (p.is_composite or p.is_borderline) \
                and feedback is None:
            case.log("S2: rule call is unambiguous; adjudication layer skipped")
            rule_h.replan_round = round_index
            return rule_h

        base = case.rule_hypothesis.value if case.rule_hypothesis else rule_h
        h, meta = s2_llm.adjudicate(d, p, base, self.wl, llm_cfg,
                                    feedback=feedback, round_index=round_index)
        case.log(f"S2: adjudication provider={meta['provider']} "
                 f"detail={meta.get('source_detail')} verdict={h.adjudication}")
        return h

    def _replan_loop(self, case: WaferCase) -> Tuple[Hypothesis, CheckReport]:
        rcfg = self.checks_cfg["replan"]
        max_rounds = int(rcfg["max_rounds"])
        min_improve = float(rcfg["min_improvement"])

        attempts: List[Hypothesis] = []
        feedback: Optional[Dict[str, Any]] = None
        best: Optional[Tuple[Hypothesis, CheckReport]] = None
        prev_score = None

        for rd in range(max_rounds):
            h = self._propose(case, feedback, rd, attempts)
            report = run_all(h, self.model, case.wafer_map.value, self.checks_cfg,
                             secondary=self.secondary,
                             descriptors=case.descriptors.value)
            score = report.violation_score
            attempts.append(h)

            stop_reason = None
            if report.passed:
                stop_reason = "checks_passed"
            elif rd == max_rounds - 1:
                stop_reason = "max_rounds_reached"
            elif prev_score is not None and prev_score > 0:
                improvement = (prev_score - score) / prev_score
                if improvement < min_improve:
                    stop_reason = (f"stalled (improvement {improvement:.1%} < "
                                   f"{min_improve:.0%})")

            case.replan_log.append(ReplanAttempt(rd, h, report, score, stop_reason))
            if best is None or score < best[1].violation_score:
                best = (h, report)

            case.log(f"S2/checks: round {rd} violation_score={score:.3f} "
                     f"passed={report.passed} stop={stop_reason}")
            if stop_reason:
                break

            feedback = build_feedback(h, report, attempts)
            prev_score = score

        return best  # type: ignore[return-value]

    # -- full run ---------------------------------------------------------- #

    def run_case(self, case: WaferCase) -> WaferCase:
        # S1
        s1.run(case)

        # S2 + physics checks + replanning
        h, report = self._replan_loop(case)
        case.check_report = report

        # Physics gate
        approval = request_approval(case, h, report,
                                    self.cfg["approval"]["mode"], str(self.out_dir))
        case.approval = approval
        case.log(f"gate: {approval.decision} ({approval.comment})")

        if approval.decision in ("rejected", "escalated"):
            case.escalated = True
            case.hypothesis = Tagged(h, Provenance.ASSUMED,
                                     f"NOT approved ({approval.decision}); "
                                     f"downstream stages skipped")
            self._write_log(case)
            return case

        case.hypothesis = Tagged(h, Provenance.ASSUMED,
                                 f"human-gated assumption ({approval.decision})")

        # S3
        spec = self.checks_cfg["device_spec"]
        wmap = case.wafer_map.value
        mask = wmap >= 0
        fields = variation_fields(h, wmap.shape[0])
        resp = self.model.response(fields, spec, knobs=None, mask=mask)
        case.device_response = Tagged(resp, Provenance.PREDICTED,
                                      f"device model '{resp.model_name}' applied to "
                                      f"the approved variation field")
        case.log(f"S3: out_of_spec_rate={resp.out_of_spec_rate:.3f} "
                 f"vth_std={resp.stats['vth_std']:.4f}")

        # S4
        comp = optimize(h, self.model, wmap, spec, self.cfg["optimizer"])
        case.compensation = Tagged(comp, Provenance.OPTIMIZED,
                                   "robust design point minimising out-of-spec rate")
        case.log(f"S4: out_of_spec {comp.baseline_out_of_spec:.3f} -> "
                 f"{comp.optimized_out_of_spec:.3f} "
                 f"({comp.improvement:+.1%}) in {comp.n_evaluations} evaluations")

        self._write_log(case)
        return case

    def _write_log(self, case: WaferCase) -> None:
        path = self.log_dir / f"{case.case_id}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"case_id": case.case_id,
                       "replan_log": [a.to_dict() for a in case.replan_log],
                       "approval": case.approval.to_dict() if case.approval else None,
                       "escalated": case.escalated,
                       "notes": case.notes},
                      fh, indent=2, ensure_ascii=False)
