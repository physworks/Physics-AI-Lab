# 13 — From measurement to circuit

Three stages of one question: **what has to be true for a compact model to be
worth using?**

Project 12 built a compact model and showed that fitting a curve well and
determining the parameters are different things. This project takes the next
three steps of the same loop.

| stage | question | state |
|---|---|---|
| **1 · design** | Which bias points should be measured so the extracted parameters are worth trusting? | **released** (`stage1_design/`) |
| **2 · corners** | How does device-to-device spread become a corner and statistical model, and do those corners actually bound circuit behaviour? | **released** (`stage2_corners/`) |
| **3 · circuit** | Does the model survive a circuit simulator — convergence, continuity, no negative conductance? | planned |

Each stage ships on its own and is tagged. The stages share a device model, so
a later stage consumes the earlier one's output rather than restating it.

## Why these three

A compact model is handed between people. Someone measures, someone extracts,
someone builds corners, and someone runs it in a circuit. Each handover is a
place the model can be quietly wrong while every individual step looks fine:
a parameter that fit well but was never determined, a corner that bounds the
parameters but not the circuit, a model that is accurate and will not converge.

## Stage 1 — results

Full report: [`stage1_design/outputs/report.md`](stage1_design/outputs/report.md).

- Choosing bias points by information rather than evenly **cut held-out
  prediction error by about 2.9×** at the same number of measurements.
- The textbook D-optimal design, which assumes uniform measurement noise, is
  **worse than the floor-aware one**: it concentrates points in deep
  subthreshold, where sensitivity is highest and precision is lowest.
- The decision that mattered most was not which criterion to optimise. It was
  whether to estimate an undeterminable parameter or fix it at its nominal —
  worth a 32% improvement or a 2× degradation depending on how well that
  nominal is known. The break-even falls **between 20% and 30% nominal
  error**, and both hand-written rules put their threshold elsewhere — each
  is wrong on 3 of the 8 cases, in opposite directions.
- **The LLM arm did not beat either rule.** Across three prompt and payload
  conditions and six sweeps it tied the stronger rule at 3 of 8 and never
  won. Every condition failed identically — fixing the parameter where it
  should have estimated it, at 30% nominal error and above. Neither rewriting
  the prompt nor supplying the decisive evidence in structured form moved
  that boundary, and the three conditions are statistically
  indistinguishable at the repeat count used.

## Stage 2 — results

Full report: [`stage2_corners/outputs/report.md`](stage2_corners/outputs/report.md).

- **Process correlation and estimation correlation have opposite signs** for
  `vth0`/`mu0` (−0.33 against +0.90), and 12 of 21 pairs disagree. Stage 1
  measured the second; a corner model needs the first. The process matrix here
  is derived from physical causes rather than written down.
- **Subtracting the extraction noise works only for parameters that were
  identifiable.** It recovers the true spread within ~10% for five of seven;
  `theta` and `vsat` — the two stage 1 flagged — come in 9–10× inflated and no
  linear correction rescues them. An unidentifiable parameter's observed
  spread is measurement noise, and belongs fixed at nominal rather than given
  a corner.
- **2 corners beat 128.** Classic independent box corners cover the population
  at +50 to +81% wasted margin; targeting the k-sigma ellipsoid for a named
  metric covers it at about 0% using two simulations.

## Honest limitations

- Synthetic devices with known ground truth throughout. No measured silicon.
- The model is the EKV-form implementation from project 12. It is **not**
  BSIM, BSIM-CMG, or any industry-standard compact model.
- Designs are built at believed parameter values, not true ones. Sequential
  re-design would close that gap and is not implemented.
- The committed stage-1 results have a **rule fallback** in the agent row: the
  development sandbox could not reach an LLM API. `scripts/run_llm_arm.py`
  fills that row in, and the report says so until it does.

## Layout

```
13_measurement-to-circuit/
  stage1_design/        optimal measurement design      (released)
  stage2_corners/       corner and statistical models   (planned)
  stage3_circuit/       circuit-level model validation  (planned)
```

Requires project `12_compact-model-extraction` alongside it, or `CMEXT_PATH`
pointing at it.
