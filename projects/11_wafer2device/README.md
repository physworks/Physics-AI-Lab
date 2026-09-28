# wafer2device

A wafer-map → process-hypothesis → device-response → compensation pipeline with a
**physics gate**: the hypothesis stage proposes, the physics checks judge, and a
human approves before anything downstream runs.

The interesting question this repository is built around is not "can a model map
a wafer pattern to a process cause" — a rule table does that. It is **where a
language model earns its place next to a rule table, and how you keep it from
quietly widening the physics search space when you let it in.**

---

## The loop

```
 S1 wafer map ──► spatial descriptors ──► pattern call (rule)
                          │
 S2                       ▼
        rule hypothesis ──► LLM adjudication (agree / refine / dispute)
                          │
        physics checks ───┤ fail ──► structured feedback ──► replan (max 3)
                          │ pass
 GATE   ◄─────────────────┘   human approval (strict / interactive / auto)
                          │
 S3 device model ─────────► per-die Vth / Ion / Ioff distribution
 S4 robust optimisation ──► design knobs minimising out-of-spec rate
 S5 report ───────────────► separated by provenance
```

Stage order is fixed, so every run is reproducible and every failure is
attributable. The autonomy lives in exactly one place: when the physics checks
fail, S2 is asked to revise using structured failure feedback.

---

## Three design decisions worth explaining

**1. The hypothesis stage proposes; it does not decide.**
Mapping a spatial defect signature to a process variation is a *physical
assumption*. Every hypothesis is validated against a whitelist
(`configs/whitelist.yaml`), checked against physics (`configs/checks.yaml`), and
written to an assumption card that a human approves before the device model ever
sees it. An LLM cannot change what counts as physically valid, and it cannot
bypass the gate.

**2. The rule engine is a strong control arm, not a strawman.**
It separates an edge ring from a radial gradient using the *shape* of the radial
profile (a ring is a step, a gradient is linear), not just the edge/centre ratio.
It references the mid annulus so that a ring and a centre spot on the same wafer
do not cancel out. It replans too. If the adjudication layer beats it, that is a
real result.

**3. A failed case is a normal outcome.**
When the replanning budget runs out, the case is escalated to the engineer and
the device and compensation stages **do not run**. A pipeline that always
produces an answer is not a pipeline you can trust with a wafer.

---

## What the comparison actually shows

`python experiments/compare_rule_vs_llm.py` (24 synthetic cases, ground truth
known):

| case kind | metric | rule | adjudicated |
|---|---|---|---|
| simple (12) | param F1 | 1.000 | 1.000 |
| simple (12) | escalation rate | 0% | 0% |
| simple (12) | out-of-spec reduction | 60.6% | 60.6% |
| **composite (6)** | **param F1** | **0.667** | **0.833** |
| **composite (6)** | **escalation rate** | **33%** | **0%** |
| **composite (6)** | **out-of-spec reduction** | **14.0%** | **28.6%** |
| composite (6) | magnitude MAE | **0.0178** | 0.0247 |
| borderline (6) | param F1 | 1.000 | 1.000 |
| borderline (6) | escalation rate | 50% | 50% |

Read it honestly:

- On **simple** patterns the adjudication layer adds nothing. It agrees with the
  rules, and it should.
- On **composite** patterns — two causes on one wafer — the rule engine is
  structurally unable to apportion a signature between them, and that is where
  the gain is.
- On **borderline** cases the current adjudicator only lowers confidence, which
  does not change a physics-check outcome. **No gain yet.** Lowered confidence
  reaches the human at the gate, which is useful, but it is not measured here.
- Even where it wins, it does not win on everything: on composite cases the rule
  engine's surviving terms are **closer in magnitude** (MAE 0.0178 vs 0.0247).
  The adjudicator finds the right *causes* and sizes them less precisely.

Those last two rows are the kind of result worth keeping in a README. An earlier
version of this table showed a much larger gain (escalation 50%→0%, out-of-spec
5.0%→56.6%) — until a defect was found in the rule engine's magnitude formula,
which had been left referencing the edge/centre ratio after the scoring moved to
the mid annulus. On exactly the composite cases the two disagreed, the rule
proposed a zero-magnitude variation: a hypothesis naming a cause while claiming
nothing varies. Fixing the control arm cut the measured gain roughly in half.
The corrected numbers are the ones above.

---

## Honesty about the data

| stage | provenance | meaning |
|---|---|---|
| wafer map | `synthetic` / `measured` | generated with known ground truth, or a real WM-811K map |
| hypothesis | `assumed` | a physical assumption, human-approved, **never a measurement** |
| device response | `predicted` | model output |
| compensation | `optimized` | optimiser output |

These tags are enforced in `contracts.py` and drive the report layout, so a
reader can always see which numbers are observations and which are assumptions.

**The WM-811K dataset carries no process ground truth.** Nothing in it says what
caused a given pattern. The pattern → process mapping is therefore an assumption
in every case, which is exactly why it sits behind a gate. Scoring only happens
on synthetic cases, where the ground truth is known by construction.

WM-811K is not redistributed here. Drop `LSWMD.pkl` into `data/` and set
`data.source: wm811k` to run against real maps.

---

## The offline adjudicator is not an LLM

`hypothesis.llm.provider` accepts:

- `openai` — a real API call (needs `allow_network: true` and `OPENAI_API_KEY`)
- `cache` — replay of recorded responses, so a reviewer reproduces a run exactly
- `stub` — a **deterministic reference adjudicator, explicitly not a language
  model.** It hand-implements the two behaviours the rules lack (composite
  decomposition, replanning under feedback) so that the pipeline and its tests
  run with no network at all.

Anything the stub produces is tagged `stub_adjudicator`, and the comparison
experiment labels that arm "reference adjudicator (NOT an LLM)". The numbers in
the table above come from the stub arm; a live-API run is one flag away
(`--arm openai`) and would replace them.

The experiment records the source of every hypothesis it generates and prints a
provenance check. If an arm's API calls all fail, the pipeline falls back to the
rule engine by design — which would make both columns describe the same system
while still looking like a comparison. That case is detected, stated in the
report, and exits non-zero rather than being published as a result.

---

## Quickstart

```bash
pip install -r requirements.txt

python run_pipeline.py                        # rule path, strict gate
python scripts/run_llm_arm.py                 # live LLM arm (needs OPENAI_API_KEY)
python run_pipeline.py --provider stub        # with the adjudication layer
python run_pipeline.py --approval interactive # review each assumption card by hand

python tests/test_pipeline.py                 # 13 tests, no pytest required
python experiments/compare_rule_vs_llm.py     # the comparison table
python scripts/plot_case.py --case edge_ring_plus_center_00 --provider stub
```

Outputs land in `outputs/` (per-case reports, assumption cards, figures) and
`logs/replan/` (the full replanning audit trail, one JSON per case).

---

## Plugging in the TCAD surrogate

`S3` talks to a `DeviceModel` interface. The analytic model is the first
implementation; the surrogate is the second, and nothing else changes:

```bash
python scripts/fit_surrogate.py --csv my_tcad_sweeps.csv
# then set device_model.secondary: surrogate in configs/pipeline.yaml
```

The two are **not interchangeable peers** — they are a precision ladder. The
analytic model is for fast screening (trends guaranteed, absolute values
first-order); the surrogate is for precise evaluation. Whenever both are
available, `physics_checks.model_agreement` compares their *trends* and fails if
they disagree, so a mismatch surfaces as a check failure instead of a silent
difference in the optimisation result.

`--demo` fits the surface against the analytic model instead of TCAD data. It
exercises the agreement check end to end and is labelled in the artefact so it
cannot be mistaken for a TCAD-trained surrogate.

---

## Where AI was used, and where it was not

Used: implementation of individual modules, boilerplate, and the adjudication
layer itself.

**Not used:** the whitelist, the physics check criteria, and the spec limits.
Those define what counts as a valid answer. A system that lets the model set its
own validation criteria has no validation — so the thresholds in
`configs/checks.yaml` are the engineer's, and the model works inside them.

---

## Limitations

- Device physics is first-order analytic (long-channel Vth, subthreshold Ioff
  via SS, Ion via overdrive). Adequate for showing the loop, not for a real
  technology node.
- The pattern → process mapping is an assumption throughout; on real wafers
  nothing here proves causality.
- Synthetic scenarios are generated from the same spatial-form library the
  hypothesis stage draws from. That makes the scoring fair between arms but
  easier than reality, where the true variation may have no matching form.
- The borderline case is the open problem: lowered confidence reaches the human
  but does not yet change a physics outcome.

## Layout

```
wafer2device/
  contracts.py        WaferCase, provenance tags, check and replan records
  orchestrator.py     fixed DAG + bounded replanning loop
  gate.py             assumption card + approval modes
  s1_pattern/         wafer data sources, descriptors, rule classification
  s2_hypothesis/      whitelist schema, rule engine, adjudication layer
  s3_device/          DeviceModel interface, analytic model, surrogate slot
  s4_compensate/      GP-based robust optimisation
  physics_checks/     range, monotonicity, limit case, pattern consistency,
                      model agreement, and the structured failure feedback
  s5_report/          provenance-separated markdown reports
configs/              whitelist, check criteria, pipeline settings
experiments/          rule vs adjudicated comparison
scripts/              surrogate fitting, per-case figures
tests/                13 tests covering the invariants
```
