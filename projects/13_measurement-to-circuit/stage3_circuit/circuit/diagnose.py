"""Diagnosing a failed solve: two rules and a model, on identical evidence.

Stage 1 measured an LLM on a quantitative threshold judgement and it never
beat a rule, failing in the same direction every time across three prompt
conditions at fifteen repeats.  **That lowers the prior here, and the prior is
written down before the measurement rather than after.**  What is different is
the shape of the judgement: a discrete classification from structured
evidence, with an answer key from fault injection.

Abstention is allowed and measured.  Accuracy on its own rewards confident
guessing, and stage 1 produced a condition that was perfectly self-consistent
and systematically wrong.  Three of the injected cases are genuinely
undecidable from the trace -- a discontinuity the solve never crosses leaves
no symptom at all, so those traces are identical to a healthy one.  A
diagnoser that abstains there is behaving better than one that guesses the
most common class, and only reporting both numbers shows the difference.

Two rule variants are provided so the comparison is fair: one that always
answers, as a rule author would naturally write it, and one that abstains when
the evidence fits more than one class.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List

from .faults import FAULT_CLASSES

ABSTAIN = "abstain"
ANSWERS = tuple(FAULT_CLASSES) + (ABSTAIN,)

PROMPT = """\
You diagnose why a DC circuit solve behaved the way it did, from the solver's \
trace. You do not run anything; the trace is all there is.

Answer with exactly one of:
  none                      converged and nothing is wrong
  c1_kink                   the device current is discontinuous, so Newton is \
given a direction that does not lead to the root and chatters
  negative_gds              output conductance is negative hard enough to \
flip the node conductance
  truncated_inner_loop      the device's internal series-resistance solve was \
cut short, so its current and its derivatives describe different devices
  bad_initial_guess         no defect in the circuit; the starting point was \
simply too far away
  singular_operating_point  no solution exists at this bias
  abstain                   the trace is consistent with more than one of the \
above and does not single one out

What the fields mean. `converged` and `iterations` are the outer Newton. \
`retry_from_default_converged` is the same circuit solved again from the \
default starting point. `started_far_from_default` says the first attempt was \
given an unusual start. `device_state` carries each device's terminal \
voltages, current, `gm`, `gds`, and whether its internal series-resistance \
loop converged. `stalled`, `diverging` and `oscillating` describe the shape of \
the residual history.

Abstain when you mean it. A trace with no anomalies is consistent both with a \
healthy circuit and with a defect the solve never excited, and guessing the \
commoner of the two is worth less than saying so.

Reply with JSON only, no prose and no code fence:
{"diagnosis": "<one of the above>", "rationale": "one sentence"}"""


def _device(trace: Dict) -> Dict:
    ds = trace.get("device_state") or []
    return ds[0] if ds else {}


def rule_diagnose(trace: Dict, allow_abstain: bool = False) -> str:
    """Heuristics over the residual pattern and the device state.

    Order matters: the specific anomalies are tested before the shape of the
    residual history, because a diverging solve produces stalling and
    oscillation as side effects of whatever actually went wrong.
    """
    d = _device(trace)
    converged = bool(trace.get("converged"))
    retry = bool(trace.get("retry_from_default_converged"))
    gds = float(d.get("gds", 0.0))
    inner_ok = bool(d.get("inner_converged", True))
    gm = float(d.get("gm", 1.0))

    if gds < 0.0:
        return "negative_gds"

    if converged and not inner_ok:
        return "truncated_inner_loop"

    if not converged and retry:
        return "bad_initial_guess"

    if not converged:
        if trace.get("oscillating"):
            return "c1_kink"
        if not inner_ok:
            return "truncated_inner_loop"
        return "singular_operating_point"

    # Converged, no anomaly.  This is consistent with a healthy circuit and
    # with a discontinuity the solve never crossed; the evidence does not
    # separate them.
    if gm == 0.0:
        return "c1_kink"
    return ABSTAIN if allow_abstain else "none"


@dataclass
class Diagnosis:
    case: str
    truth: str
    answer: str
    source: str
    rationale: str = ""

    @property
    def correct(self) -> bool:
        return self.answer == self.truth

    @property
    def abstained(self) -> bool:
        return self.answer == ABSTAIN


@dataclass
class ArmResult:
    name: str
    diagnoses: List[Diagnosis] = field(default_factory=list)

    def summary(self, undecidable: List[str] | None = None) -> Dict:
        und = set(undecidable or [])
        n = len(self.diagnoses)
        answered = [d for d in self.diagnoses if not d.abstained]
        correct = [d for d in answered if d.correct]
        decidable = [d for d in self.diagnoses if d.case not in und]
        dec_correct = [d for d in decidable if d.correct]
        abst_on_und = [d for d in self.diagnoses
                       if d.abstained and d.case in und]
        return {
            "arm": self.name, "n": n,
            "accuracy_all": len(correct) / n if n else float("nan"),
            "n_answered": len(answered),
            "accuracy_when_answered": (len(correct) / len(answered)
                                       if answered else float("nan")),
            "abstention_rate": 1.0 - len(answered) / n if n else float("nan"),
            "accuracy_on_decidable": (len(dec_correct) / len(decidable)
                                      if decidable else float("nan")),
            "abstained_on_undecidable": len(abst_on_und),
            "n_undecidable": len(und),
            "confusion": self.confusion(),
        }

    def confusion(self) -> Dict[str, Dict[str, int]]:
        m: Dict[str, Dict[str, int]] = {t: {a: 0 for a in ANSWERS}
                                        for t in FAULT_CLASSES}
        for d in self.diagnoses:
            m.setdefault(d.truth, {a: 0 for a in ANSWERS})
            m[d.truth][d.answer] = m[d.truth].get(d.answer, 0) + 1
        return m


# --------------------------------------------------------------------------- #
# The LLM arm, same shape as stage 1: schema-checked, rule fallback, provider
# inferred from the model name.
# --------------------------------------------------------------------------- #

PROVIDERS = {
    "openai": {"url": "https://api.openai.com/v1/chat/completions",
               "env": ("OPENAI_API_KEY",)},
    "anthropic": {"url": "https://api.anthropic.com/v1/messages",
                  "env": ("ANTHROPIC_API_KEY",)},
    "gemini": {"url": "https://generativelanguage.googleapis.com/v1beta/"
                      "models/{model}:generateContent",
               "env": ("GEMINI_API_KEY", "GOOGLE_API_KEY")},
}


def provider_for(model: str) -> str:
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "gemini"
    return "openai"


def api_key_for(provider: str) -> str | None:
    for name in PROVIDERS[provider]["env"]:
        v = os.environ.get(name)
        if v:
            return v
    return None


def evidence_for(trace: Dict) -> Dict:
    """What the model sees.  Symptoms only -- no field names the fault."""
    d = _device(trace)
    return {
        "converged": trace.get("converged"),
        "iterations": trace.get("iterations"),
        "final_residual": trace.get("final_residual"),
        "residual_history": [round(float(v), 12)
                             for v in trace.get("residual_history", [])][:25],
        "stalled": trace.get("stalled"),
        "diverging": trace.get("diverging"),
        "oscillating": trace.get("oscillating"),
        "damped_steps": trace.get("damped_steps"),
        "max_jacobian_cond": trace.get("max_jacobian_cond"),
        "retry_from_default_converged": trace.get(
            "retry_from_default_converged"),
        "started_far_from_default": trace.get("started_far_from_default"),
        "device_state": [{k: v for k, v in dd.items()}
                         for dd in (trace.get("device_state") or [])],
    }


def _call(payload: Dict, model: str, key: str, provider: str,
          timeout: float = 60.0) -> str:
    user = json.dumps(payload, ensure_ascii=False)
    spec = PROVIDERS[provider]
    if provider == "gemini":
        body = {"systemInstruction": {"parts": [{"text": PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0,
                                     "responseMimeType": "application/json"}}
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    elif provider == "anthropic":
        body = {"model": model, "max_tokens": 512, "temperature": 0,
                "system": PROMPT,
                "messages": [{"role": "user", "content": user}]}
        headers = {"Content-Type": "application/json", "x-api-key": key,
                   "anthropic-version": "2023-06-01"}
    else:
        body = {"model": model, "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": PROMPT},
                             {"role": "user", "content": user}]}
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {key}"}

    req = urllib.request.Request(spec["url"].format(model=model),
                                 data=json.dumps(body).encode("utf-8"),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if provider == "gemini":
        return "".join(p.get("text", "")
                       for p in data["candidates"][0]["content"]["parts"])
    if provider == "anthropic":
        return "".join(b.get("text", "") for b in data["content"]
                       if b.get("type", "text") == "text")
    return data["choices"][0]["message"]["content"]


def llm_diagnose(trace: Dict, model: str = "gpt-4o-mini",
                 provider: str | None = None, transport=None
                 ) -> tuple[str, str, bool]:
    """Returns (answer, rationale, reached_model); falls back to the rule."""
    provider = provider or provider_for(model)
    key = api_key_for(provider)
    if transport is None and not key:
        return rule_diagnose(trace), "rule fallback: no API key", False

    payload = evidence_for(trace)
    try:
        raw = transport(payload) if transport else _call(payload, model, key,
                                                         provider)
        parsed = json.loads(raw)
        answer = str(parsed.get("diagnosis", "")).strip()
        if answer not in ANSWERS:
            return (rule_diagnose(trace),
                    f"rule fallback: {answer!r} is not an allowed answer",
                    False)
        return answer, str(parsed.get("rationale", ""))[:300], True
    except (json.JSONDecodeError, urllib.error.URLError,
            urllib.error.HTTPError, OSError, KeyError, TimeoutError) as exc:
        return rule_diagnose(trace), f"rule fallback: {exc}", False
