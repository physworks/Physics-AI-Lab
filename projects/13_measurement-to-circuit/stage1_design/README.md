# Stage 1 — optimal measurement design

Which bias points should be measured so that the parameters extracted from
them are worth trusting?

```bash
pip install -r requirements.txt
python run_design.py              # writes outputs/
python -m pytest tests/ -q        # 23 tests
```

Needs `12_compact-model-extraction` beside this project, or `CMEXT_PATH` set
to it. The device model and the extractor come from there deliberately — a
second, subtly different copy would make every comparison meaningless.

## The question behind the question

Fisher information says how much a measurement tells you about a parameter.
Maximising it over a set of candidate bias points is a solved problem: the
log-determinant of a Fisher matrix is submodular, so greedy selection carries
a (1 − 1/e) guarantee and in practice lands very close to optimal.

**So "an agent beats greedy at D-optimal design" is not a claim worth making,
and this project does not make it.** The selector is the same constrained
greedy in every arm. What differs is the objective it is handed, and the
objective is a judgement, not a calculation.

## What was measured

**1 · Choosing points by information is worth about 2.9×** in held-out
prediction error against an evenly spaced grid, at the same 12 measurements.
The evaluation metric is deliberately outside every arm's objective: fit at
the designed points, then predict current on a dense grid no design ever saw.

**2 · The textbook design is the wrong one.** D-optimal under a uniform-noise
assumption puts points in deep subthreshold, where the Jacobian is largest.
That is also where the current approaches the instrument's noise floor and
relative precision collapses. Weighting the Fisher matrix by the real,
current-dependent noise moves the points and improves held-out error further.

> This also explains an anomaly in project 12. Its `measure()` has a noise
> floor but its `analyse()` assumes uniform noise, and its predicted 1σ for
> `ss` was 1.54× optimistic against Monte Carlo. Floor-aware weighting brings
> that to 1.36×; the rest is genuine nonlinearity. Project 12's report rounded
> to two decimals, which hid the discrepancy entirely.

**3 · The decision that dominated everything** was not the criterion. `theta`
has about 1/87 of `vth0`'s sensitivity and no measurement plan recovers it.
Fixing it removes a nuisance direction and improves every other parameter —
*if* the value it is fixed at is close enough. Fixed at a stale default it
injects a bias no plan can undo.

| nominal error on `theta` | better choice |
|---|---|
| ≤ 20% | fix it (up to 32% better) |
| ≥ 30% | estimate it (up to 2.0× worse if fixed) |

The break-even falls **between 20% and 30%**. The sensitivity-only rule never
looks at trust and is wrong above it; the provenance-aware rule guesses a 5%
threshold and is wrong between 10% and 20%. Each is wrong on 3 of 8 cases, in
opposite directions, and neither author knew where the boundary was.

## What the agent is allowed to do

It chooses **which parameters the design should determine** and **which
criterion scores them**. That is all.

It does not compute anything — every Fisher matrix, score and extraction runs
in this package without an API key. It does not select bias points; the same
selector serves every arm, so a win is attributable to the objective and not
to a better search. It cannot touch the stress limits, the settling limit or
the specification points, and anything outside the schema in
`objective.validate` is refused rather than repaired, falling back to the
rule.

If no key is present the arm runs the rule and labels itself a fallback.
`run_design.py --require-llm` exits 3 when the agent arm was asked for and
never reached a model, because a run where every call fell back is not
evidence about an agent.

Any of three providers works — the arm needs a model that returns JSON and
nothing else. The provider is inferred from the model name, overridable with
`--provider`:

| provider | env var | example model |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` |
| Google | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-3.5-flash-lite` |

```bash
export GEMINI_API_KEY=...                       # or put it in .env
python scripts/run_llm_arm.py --model gemini-3.5-flash-lite --repeats 3
```

24 calls per run of three repeats. The key may instead live in a `.env` file
beside the project — gitignored, because this repository is public — and a
real environment variable always wins over it. The key is sent as a header on
every provider, never as a URL query parameter, so it does not reach logs or
shell history.

Model names age faster than this code. If a run fails with a 404, ask the
provider what the key can actually reach:

```bash
python scripts/run_llm_arm.py --provider gemini --list-models
```

### The agent arm, measured six times across three conditions

The first run got 3 of 8 wrong, fixing `theta` at every trust level -- the
same decisions as the rule that never looks at provenance. Inspecting the
request found two defects on *this* side of the experiment: the structured
`nominal_provenance` was assembled as evidence but never put in the payload,
and the system prompt described the cost of *keeping* an insensitive
parameter while never mentioning the bias that *fixing* one locks in. It
argued one side of a two-sided trade.

Both were fixed (v2), and all three conditions were then measured:

| | prompt | provenance in payload | wrong (15 repeats) |
|---|---|---|---|
| A | v1 | no | 3 / 8 |
| B | v1 | yes | 3 / 8 |
| C | v2 | yes | 3 / 8 |

Per-run self-agreement and the levels that split are in
`outputs/report.md` and in each `outputs/llm_arm_*.json`.

**The conditions are indistinguishable**, and the reading that got there was
itself a lesson. An early three-repeat sweep of condition B returned 5 of 8
and was briefly written up as evidence that unexplained evidence *hurts*. It
did not reproduce. With three repeats and a majority vote, a level where the
model sits near 50/50 resolves correctly only half the time, so 3-of-8 and
5-of-8 are equally likely outcomes of one unchanged process (p = 0.22 each) —
the claim was reading sampling noise as an effect. Re-run at 15 repeats, all
three conditions return 3 of 8. The runner now flags non-unanimous levels and
refuses to let an underpowered run be compared across conditions.

What *is* stable across every condition and every sweep:

- **The agent never beats a rule.** Its best ties `rule_with_provenance` at
  3 of 8.
- **It fails in the same place, in the same direction, every time**: at 30%
  nominal error and above it keeps choosing to fix `theta` when the measured
  answer is to estimate it. Neither a better prompt nor the decisive evidence
  in structured form moved that boundary.
- High self-agreement belongs to conditions that are systematically wrong.
  Consistency here measures repeatability, not correctness.

To claim a prompt effect at all, run each condition with many more repeats:

```bash
python scripts/run_llm_arm.py --prompt-version 1                 --repeats 15
python scripts/run_llm_arm.py --prompt-version 1 --provenance on --repeats 15
python scripts/run_llm_arm.py --prompt-version 2                 --repeats 15
```

The same discipline applied twice to the rule baselines in project 11 applies
here, and it cut against the agent both times: once by exposing that the
prompt was the problem, and once by withdrawing a finding that did not
replicate.

## What this does not show

- **The agent arm in the committed results is a rule fallback.** The sandbox
  this was built in cannot reach `api.openai.com`. Run
  `scripts/run_llm_arm.py` to fill it in; it scores the model against the
  measured boundary and repeats the sweep to check the answer is stable.
- A c-optimal design on `vth0` beat D-optimal on `vth0`'s uncertainty by 4.7%
  over 60 noise realisations. The sampling error of a standard deviation from
  60 samples is about 9%, so this is **not established** and is reported that
  way. (c-optimal with all seven parameters free is much *worse* than D —
  precision cannot be bought on one parameter while its 0.94-correlated
  partner floats.)
- The engineer heuristic came out worse than the uniform grid here. That is
  one hand-written allocation on one synthetic device, not a finding about
  how engineers choose points.
- Everything is synthetic, with known ground truth. No measured silicon.

## Layout

```
oed/
  device.py       bridge to project 12's model and extractor
  noise.py        current-dependent instrument noise
  fisher.py       weighted Fisher information, D/A/E/c criteria, row cache
  constraints.py  candidate pool, stress and settling limits, spec points
  select.py       constrained greedy, and the uniform / heuristic baselines
  objective.py    the decision under study, and the two rule baselines
  agent.py        the LLM arm, its schema, and its fallback
  pipeline.py     fixed stage order with one bounded replan
  evaluate.py     held-out prediction error and Monte Carlo verification
  scenarios.py    the break-even sweep that defines the right answer
```
