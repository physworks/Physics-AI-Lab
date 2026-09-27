"""S2 (LLM path) -- adjudicate the rule hypothesis, never replace it blindly.

The LLM receives the descriptors *and* the rule hypothesis, and must answer with
one of three verdicts:

    agree   -- the rule hypothesis stands
    refine  -- same causes, adjusted magnitudes / confidence
    dispute -- the rule hypothesis is wrong; an alternative is supplied

This mirrors a review layer rather than an oracle: the deterministic path stays
the default and the model has to argue its way past it.

Three providers are available, and which one ran is always recorded in the
output so results can never be mistaken for one another:

``openai``    real API call (requires ``allow_network: true`` and OPENAI_API_KEY)
``cache``     replay of a previously recorded response, keyed by case content
``stub``      a deterministic *reference adjudicator*.  IT IS NOT AN LLM.  It
              exists so the pipeline and its tests run with no network at all.
              Anything it produces is tagged ``stub_adjudicator`` and the
              comparison experiment refuses to score it as an LLM arm.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..contracts import (Hypothesis, PatternCall, SpatialDescriptors,
                         VariationTerm)
from .rules import PATTERN_TO_TERM
from .schema import Whitelist, coerce_terms, validate_hypothesis

SYSTEM_PROMPT = """You are a semiconductor yield-analysis assistant.

You are given quantitative spatial descriptors extracted from a wafer map and a
hypothesis produced by a deterministic rule engine. Your job is to adjudicate
that hypothesis, not to replace it by default.

Rules you must follow:
1. Use ONLY the parameters and spatial forms in the allowed vocabulary below.
2. Reply with a single JSON object and nothing else.
3. Ground every claim in the numeric descriptors you were given. Do not invent
   measurements.
4. Prefer "agree" unless the descriptors give a concrete reason not to.

Response schema:
{
  "adjudication": "agree" | "refine" | "dispute",
  "terms": [{"param": str, "spatial_form": str, "magnitude": float,
             "confidence": float}],
  "rationale": str
}
For "agree", echo the rule hypothesis terms unchanged."""


def build_user_prompt(d: SpatialDescriptors, p: PatternCall,
                      rule_h: Hypothesis, wl: Whitelist,
                      feedback: Optional[Dict[str, Any]] = None) -> str:
    parts = [
        "Allowed vocabulary:",
        wl.describe(),
        "",
        "Spatial descriptors:",
        json.dumps(d.to_dict(), indent=2),
        "",
        "Rule engine pattern call:",
        json.dumps(p.to_dict(), indent=2),
        "",
        "Rule engine hypothesis:",
        json.dumps(rule_h.to_dict(), indent=2),
    ]
    if feedback:
        parts += [
            "",
            "A previous hypothesis FAILED physics validation. Revise it.",
            json.dumps(feedback, indent=2),
            "",
            "Do not repeat any hypothesis listed in previous_attempts.",
        ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Provider dispatch
# --------------------------------------------------------------------------- #

def adjudicate(d: SpatialDescriptors, p: PatternCall, rule_h: Hypothesis,
               wl: Whitelist, cfg: Dict[str, Any],
               feedback: Optional[Dict[str, Any]] = None,
               round_index: int = 0) -> Tuple[Hypothesis, Dict[str, Any]]:
    """Return (hypothesis, meta).  Falls back to the rule hypothesis on any
    failure -- an unavailable model must never stop the line."""
    provider = str(cfg.get("provider", "cache"))
    prompt = build_user_prompt(d, p, rule_h, wl, feedback)
    key = _cache_key(prompt, cfg.get("model", ""))
    cache_dir = Path(cfg.get("cache_path", "llm_cache"))

    meta: Dict[str, Any] = {"provider": provider, "cache_key": key,
                            "round_index": round_index}

    raw: Optional[Dict[str, Any]] = None
    if provider in ("cache", "openai"):
        raw = _read_cache(cache_dir, key)
        if raw is not None:
            meta["source_detail"] = "cache_hit"

    if raw is None and provider == "openai":
        if not cfg.get("allow_network", False):
            meta["source_detail"] = "network_disabled"
        else:
            raw, err = _call_openai(prompt, cfg)
            meta["source_detail"] = "api_call" if raw else f"api_error: {err}"
            if raw is not None:
                _write_cache(cache_dir, key, raw, prompt)

    if raw is None and provider == "stub":
        raw = _stub_adjudicator(d, p, rule_h, wl, feedback)
        meta["source_detail"] = "stub_adjudicator (NOT an LLM)"

    if raw is None:
        meta.setdefault("source_detail", "unavailable")
        fallback = Hypothesis(terms=list(rule_h.terms),
                              rationale=rule_h.rationale + " [LLM unavailable; "
                                                           "rule hypothesis kept]",
                              source="rule_fallback", adjudication=None,
                              replan_round=round_index)
        return fallback, meta

    verdict = str(raw.get("adjudication", "agree")).lower()
    terms = coerce_terms(raw.get("terms", []), wl)
    if not terms:
        terms = list(rule_h.terms)
        verdict = "agree"
        meta["coercion"] = "no valid terms survived whitelist validation"

    source = {"stub": "stub_adjudicator"}.get(provider, "llm")
    if source == "llm":
        source = {"agree": "llm", "refine": "llm_refine",
                  "dispute": "llm_dispute"}.get(verdict, "llm")

    h = Hypothesis(terms=terms, rationale=str(raw.get("rationale", "")),
                   source=source, adjudication=verdict, replan_round=round_index)
    ok, errors = validate_hypothesis(h, wl)
    if not ok:
        meta["validation_errors"] = errors
        return Hypothesis(terms=list(rule_h.terms),
                          rationale=rule_h.rationale + f" [adjudication rejected: {errors}]",
                          source="rule_fallback", replan_round=round_index), meta
    return h, meta


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

def _cache_key(prompt: str, model: str) -> str:
    return hashlib.sha256((model + "\n" + prompt).encode("utf-8")).hexdigest()[:20]


def _read_cache(cache_dir: Path, key: str) -> Optional[Dict[str, Any]]:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)["response"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def _write_cache(cache_dir: Path, key: str, response: Dict[str, Any],
                 prompt: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"prompt": prompt, "response": response}
    with open(cache_dir / f"{key}.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _call_openai(prompt: str, cfg: Dict[str, Any]
                 ) -> Tuple[Optional[Dict[str, Any]], str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY not set"
    try:
        from openai import OpenAI          # imported lazily; optional dependency

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=cfg.get("model", "gpt-4o-mini"),
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content), ""
    except Exception as exc:                # noqa: BLE001 - provider-agnostic
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Offline reference adjudicator -- explicitly NOT a language model
# --------------------------------------------------------------------------- #

def _stub_adjudicator(d: SpatialDescriptors, p: PatternCall, rule_h: Hypothesis,
                      wl: Whitelist, feedback: Optional[Dict[str, Any]]
                      ) -> Dict[str, Any]:
    """Deterministic stand-in that exercises the same three code paths.

    It implements, by hand, the two behaviours the rule engine structurally
    lacks: composite decomposition and replanning under failure feedback.  Its
    purpose is to keep the pipeline runnable and testable offline; it is not a
    substitute for the model and is never scored as one.
    """
    fired = [k for k, v in p.scores.items() if v >= 0.5 and k in PATTERN_TO_TERM]

    if feedback:
        terms: List[Dict[str, Any]] = []
        tried = {(a.get("param"), a.get("spatial_form"))
                 for att in feedback.get("previous_attempts", [])
                 for a in att.get("terms", [])}
        for t in rule_h.terms:
            spec = wl.parameters[t.param]
            scale = 0.6 if any("decrease" in str(f.get("direction_hint", ""))
                               for f in feedback.get("failures", [])) else 1.4
            terms.append({"param": t.param, "spatial_form": t.spatial_form,
                          "magnitude": float(np.clip(t.magnitude * scale,
                                                     spec["min"], spec["max"])),
                          "confidence": 0.4, "angle_deg": t.angle_deg})
        # If the same parameter has already failed twice, blame a second cause.
        if len(tried) >= 1 and len(fired) >= 2:
            second = [f for f in fired if PATTERN_TO_TERM[f]["param"]
                      not in {t["param"] for t in terms}]
            if second:
                m = PATTERN_TO_TERM[second[0]]
                terms.append({"param": m["param"], "spatial_form": m["spatial_form"],
                              "magnitude": 0.04, "confidence": 0.35,
                              "angle_deg": (d.dominant_angle_deg
                                            if m["spatial_form"] == "angular" else None)})
        return {"adjudication": "refine", "terms": terms,
                "rationale": "Reference adjudicator: magnitudes rescaled along the "
                             "physics-check direction hints, avoiding repeats."}

    if len(fired) >= 2:
        terms = []
        total = sum(p.scores[f] for f in fired)
        for f in fired[:wl.limits.get("max_terms", 3)]:
            m = PATTERN_TO_TERM[f]
            share = p.scores[f] / max(total, 1e-6)
            base = rule_h.terms[0].magnitude if rule_h.terms else 0.05
            spec = wl.parameters[m["param"]]
            terms.append({"param": m["param"], "spatial_form": m["spatial_form"],
                          "magnitude": float(np.clip(max(base, 0.03) * (0.6 + share),
                                                     spec["min"], spec["max"])),
                          "confidence": float(0.5 + 0.4 * share),
                          "angle_deg": (d.dominant_angle_deg
                                        if m["spatial_form"] == "angular" else None)})
        return {"adjudication": "dispute", "terms": terms,
                "rationale": "Reference adjudicator: two pattern rules fired "
                             "simultaneously; the single-cause rule hypothesis was "
                             "decomposed into contributions weighted by rule score."}

    if p.is_borderline:
        terms = [{"param": t.param, "spatial_form": t.spatial_form,
                  "magnitude": t.magnitude, "confidence": 0.35} for t in rule_h.terms]
        return {"adjudication": "refine", "terms": terms,
                "rationale": "Reference adjudicator: scores sit inside the decision "
                             "margin; the hypothesis is kept but confidence is lowered "
                             "so the approval gate sees the ambiguity."}

    return {"adjudication": "agree",
            "terms": [t.to_dict() for t in rule_h.terms],
            "rationale": "Reference adjudicator: descriptors give no reason to "
                         "depart from the rule hypothesis."}
