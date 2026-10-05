"""The LLM arm: it chooses the objective, and nothing else.

The authority boundary is the whole design, so it is worth stating plainly.

The model **may** decide which parameters the measurement plan should try to
determine and which criterion scores them.  That is a judgement about what
the model is for, informed by evidence it is given rather than computed by it.

The model **may not**:

  * compute anything -- every Fisher matrix, score and extraction is done by
    code in this package and can be re-run without an API key;
  * select bias points -- the same constrained greedy selector serves every
    arm, so a win is attributable to the objective and not to a better search;
  * touch the constraints -- stress limits, settling limits and specification
    points live in the configuration;
  * widen its own remit -- anything outside the schema in ``objective.validate``
    is refused, and a refusal falls back to the rule arm.

If no API key is present the arm runs the rule objective and says so in its
provenance.  A run where every call fell back is not a demonstration of
anything, and ``run_design.py`` exits non-zero when the LLM arm was requested
but never actually reached a model.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .device import PARAM_NAMES
from .objective import (Objective, ObjectiveRejected, rule_objective,
                        validate)

DEFAULT_MODEL = "gpt-4o-mini"

#: Provider endpoints.  The arm needs a model that returns JSON on demand and
#: nothing else, so either provider does; use whichever key you have.
PROVIDERS = {
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "env": ("OPENAI_API_KEY",),
        "default_model": "gpt-4o-mini",
        "models_url": "https://api.openai.com/v1/models"},
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "env": ("ANTHROPIC_API_KEY",),
        "default_model": "claude-sonnet-4-5",
        "models_url": "https://api.anthropic.com/v1/models"},
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models/"
               "{model}:generateContent",
        "env": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "default_model": "gemini-3.5-flash-lite",
        "models_url":
            "https://generativelanguage.googleapis.com/v1beta/models"},
}

#: Model names go stale faster than this code does.  ``--list-models`` asks
#: the provider what the key can actually reach rather than trusting a
#: default written months ago.
_PREFIX = (("claude", "anthropic"), ("gemini", "gemini"),
           ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"),
           ("o4", "openai"))


def provider_for(model: str) -> str:
    """Infer the provider from the model name; override it if the guess is wrong."""
    m = model.lower()
    for prefix, provider in _PREFIX:
        if m.startswith(prefix):
            return provider
    return "openai"


def load_dotenv(path: str | None = None) -> None:
    """Read ``KEY=value`` lines from a .env file into the environment.

    Keeping the key in a gitignored file beats pasting it on a command line
    that lands in shell history.  Existing environment variables win, and a
    missing file is not an error.
    """
    candidates = [Path(path)] if path else [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env"]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        raw = env_path.read_bytes()
        for enc in ("utf-8-sig", "utf-16", "utf-8", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:                                    # pragma: no cover
            return
        for line in text.replace("\x00", "").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"\''))
        return


def api_key_for(provider: str) -> str | None:
    for name in PROVIDERS[provider]["env"]:
        value = os.environ.get(name)
        if value:
            return value
    return None


def key_env_names(provider: str) -> str:
    return " or ".join(PROVIDERS[provider]["env"])


def list_models(provider: str, api_key: str, timeout: float = 30.0) -> list:
    """What this key can actually reach.  Used to diagnose a stale model name."""
    spec = PROVIDERS[provider]
    headers = {"Content-Type": "application/json"}
    if provider == "anthropic":
        headers |= {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    elif provider == "gemini":
        headers["x-goog-api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(spec["models_url"], headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if provider == "gemini":
        return sorted(m["name"].removeprefix("models/")
                      for m in data.get("models", [])
                      if "generateContent" in m.get(
                          "supportedGenerationMethods", ["generateContent"]))
    return sorted(m["id"] for m in data.get("data", []))

PROMPT_VERSION = 2

#: v1 kept verbatim, because the first measured run used it and the report
#: reports both.  It describes only the cost of *keeping* an insensitive
#: parameter and never mentions the bias that fixing one locks in, so it
#: argues for one side of a two-sided trade.  The model followed it.
SYSTEM_PROMPT_V1 = """\
You choose the estimation objective for an optimal measurement design on a \
MOSFET compact model. You do not compute anything and you do not choose bias \
points; a separate optimiser does that from your objective.

Decide two things:
  include_params - the parameters the measurement plan should determine. A \
parameter left out is held fixed at its nominal value during extraction, so \
leaving one out is a claim that the data cannot determine it and that fixing \
it is acceptable for the model's stated purpose.
  criterion - how the joint uncertainty is scored:
    D  minimises the volume of the confidence region (every parameter equally)
    A  minimises average variance (robust when uncertainties are badly spread)
    E  minimises the worst-case variance
    c  minimises the variance of one named parameter (needs c_target)

Guidance. A parameter with very low sensitivity cannot be rescued by any \
measurement plan; keeping it in a D objective lets the determinant chase a \
direction carrying no information, at the expense of parameters that can be \
determined. A strong correlation is different: it often can be broken by \
measuring at different drain bias, so correlated parameters are usually worth \
keeping. Do not drop a parameter the stated purpose depends on merely because \
it is hard.

Reply with JSON only, no prose and no code fence:
{"include_params": [...], "criterion": "D"|"A"|"E"|"c", "c_target": null, \
"rationale": "one or two sentences"}"""

#: v2 states both sides of the trade and puts the provenance in the payload.
#: It deliberately does not say where the boundary falls -- that is the thing
#: being measured, and a prompt that named it would be teaching to the test.
SYSTEM_PROMPT = """\
You choose the estimation objective for an optimal measurement design on a \
MOSFET compact model. You do not compute anything and you do not choose bias \
points; a separate optimiser does that from your objective.

Decide two things:
  include_params - the parameters the measurement plan should determine.
  criterion - how the joint uncertainty is scored:
    D  minimises the volume of the confidence region (every parameter equally)
    A  minimises average variance (robust when uncertainties are badly spread)
    E  minimises the worst-case variance
    c  minimises the variance of one named parameter (needs c_target)

A parameter you leave out is HELD FIXED at its nominal value during \
extraction. That is a trade with two sides, and both are real:

  Keeping it in costs VARIANCE. An insensitive parameter adds a poorly \
constrained direction to the estimation problem; measurements are spent on \
it, and where it correlates with others it inflates their uncertainty too.

  Leaving it out costs BIAS. The model is pinned to the nominal value. If \
that nominal is wrong by some fraction, the error propagates into the other \
parameters and into the model's predictions, and no measurement plan can \
remove it. The bias grows with how wrong the nominal is.

`evidence.nominal_provenance` gives, for each parameter, where its nominal \
came from and its relative uncertainty. Weigh the bias that fixing would lock \
in against the variance that estimating would cost. Neither answer is right \
in general; it depends on the numbers you are given.

A strong correlation is a different problem from low sensitivity: it can \
often be broken by measuring at a different drain bias, so correlated \
parameters are usually worth keeping. Do not drop a parameter the stated \
purpose depends on merely because it is hard.

Reply with JSON only, no prose and no code fence:
{"include_params": [...], "criterion": "D"|"A"|"E"|"c", "c_target": null, \
"rationale": "one or two sentences"}"""


@dataclass
class AgentDecision:
    objective: Objective
    used_llm: bool
    model: str | None = None
    prompt_version: int = PROMPT_VERSION
    attempts: int = 0
    errors: List[str] = field(default_factory=list)

    def provenance(self) -> Dict:
        return {"source": "llm" if self.used_llm else "rule-fallback",
                "model": self.model, "prompt_version": self.prompt_version,
                "attempts": self.attempts,
                "errors": list(self.errors),
                "objective": self.objective.describe()}


def build_request(evidence: Dict, constraints: Dict, purpose: str,
                  prompt_version: int = PROMPT_VERSION,
                  with_provenance: bool | None = None) -> Dict:
    """Exactly what the model is shown.  Written out so a run is auditable.

    ``nominal_provenance`` is part of the v2 payload only.  v1 shipped without
    it -- the model saw the trust level solely as a clause inside the purpose
    sentence -- and reproducing v1 means reproducing that too, not just the
    prompt text.  Keeping the two coupled is what makes ``--prompt-version 1``
    an actual replay of the first measured run rather than a third condition
    wearing its name.
    """
    payload = {"model_purpose": purpose,
               "allowed_parameters": list(PARAM_NAMES),
               "allowed_criteria": ["D", "A", "E", "c"],
               "constraints": constraints,
               "evidence": {
                   "nominal_rel_sigma_percent": {
                       k: round(v * 100, 4)
                       for k, v in evidence["rel_sigma"].items()},
                   "sensitivity_rank": sorted(
                       evidence["sensitivity"],
                       key=lambda k: -evidence["sensitivity"][k]),
                   "sensitivity_relative_to_best": {
                       k: round(v / max(evidence["sensitivity"].values()
                                        or [1]), 5)
                       for k, v in evidence["sensitivity"].items()},
                   "strong_correlations": evidence["strong_correlations"],
                   "condition_number": evidence["condition_number"]}}

    include = (int(prompt_version) >= 2 if with_provenance is None
               else bool(with_provenance))
    if include:
        payload["evidence"]["nominal_provenance"] = {
            k: {"source": v.get("source", "unknown"),
                "rel_uncertainty_percent": round(
                    float(v.get("rel_uncertainty", 1.0)) * 100, 2)}
            for k, v in (evidence.get("nominal_provenance") or {}).items()}
    return payload


def system_prompt(version: int = PROMPT_VERSION) -> str:
    return {1: SYSTEM_PROMPT_V1, 2: SYSTEM_PROMPT}[int(version)]


def _call_model(payload: Dict, model: str, api_key: str, provider: str,
                timeout: float = 60.0, prompt_version: int = PROMPT_VERSION
                ) -> str:
    """One request, one JSON answer.  Every provider, same contract."""
    user = json.dumps(payload, ensure_ascii=False)
    spec = PROVIDERS[provider]
    sys_text = system_prompt(prompt_version)

    if provider == "gemini":
        body = {"systemInstruction": {"parts": [{"text": sys_text}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0,
                                     "responseMimeType": "application/json"}}
        headers = {"Content-Type": "application/json",
                   "x-goog-api-key": api_key}
    elif provider == "anthropic":
        body = {"model": model, "max_tokens": 1024, "temperature": 0,
                "system": sys_text,
                "messages": [{"role": "user", "content": user}]}
        headers = {"Content-Type": "application/json",
                   "x-api-key": api_key,
                   "anthropic-version": "2023-06-01"}
    else:
        body = {"model": model, "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": sys_text},
                             {"role": "user", "content": user}]}
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {api_key}"}

    url = spec["url"].format(model=model)
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if provider == "gemini":
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    if provider == "anthropic":
        return "".join(b.get("text", "") for b in data["content"]
                       if b.get("type", "text") == "text")
    return data["choices"][0]["message"]["content"]


def decide(evidence: Dict, constraints: Dict, nominal: Dict[str, float],
           purpose: str, model: str = DEFAULT_MODEL,
           max_attempts: int = 2, api_key: str | None = None,
           transport=None, provider: str | None = None,
           prompt_version: int = PROMPT_VERSION,
           with_provenance: bool | None = None) -> AgentDecision:
    """Ask the model for an objective; fall back to the rule on any failure."""
    provider = provider or provider_for(model)
    key = api_key if api_key is not None else api_key_for(provider)
    fallback = rule_objective(evidence, nominal)

    if transport is None and not key:
        return AgentDecision(
            objective=fallback, used_llm=False,
            prompt_version=prompt_version,
            errors=[f"{key_env_names(provider)} is not set"])

    payload = build_request(evidence, constraints, purpose, prompt_version,
                            with_provenance)
    errors: List[str] = []

    for attempt in range(1, max_attempts + 1):
        try:
            raw = (transport(payload) if transport is not None
                   else _call_model(payload, model, key, provider,
                                    prompt_version=prompt_version))
            parsed = json.loads(raw)
            objective = validate(parsed, PARAM_NAMES, nominal)
            return AgentDecision(objective=objective, used_llm=True,
                                 model=model if transport is None else "stub",
                                 prompt_version=prompt_version,
                                 attempts=attempt, errors=errors)
        except (ObjectiveRejected, json.JSONDecodeError) as exc:
            errors.append(f"attempt {attempt}: rejected: {exc}")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                KeyError, TimeoutError) as exc:
            errors.append(f"attempt {attempt}: transport: {exc}")
            break

    return AgentDecision(objective=fallback, used_llm=False, model=model,
                         prompt_version=prompt_version,
                         attempts=max_attempts, errors=errors)
