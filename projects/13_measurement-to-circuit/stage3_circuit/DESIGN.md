# Stage 3 — circuit-level model validation

**Question.** Does the model survive a circuit simulator, and do stage 2's
corners still bound behaviour once the metric is a real solve rather than a
proxy?

Scope settled up front: **DC and transient both**, and the diagnostic agent
**may abstain**.

## Job 0 — the device evaluation has to be fixed first

This is a prerequisite, not a refinement. `drain_current` in project 12 solves
series resistance by damped fixed-point iteration capped at 40 steps, and
returns the 40th value whether or not it converged.

| | iterations needed | cap |
|---|---|---|
| nominal | **39** | 40 |
| stage 2 population (3000 devices) | 38 max | 40 |
| `rs = 2000` (the model's own upper bound) | **200+** | 40 — returns unconverged |

Stage 2's results stand: nothing in that population crossed the cap. **The
margin is one iteration**, and a circuit solver explores bias points stage 2
never visited.

Two further problems matter only once derivatives are needed:

- Finite-differencing the current gives the derivative *of a truncated
  iteration*, not of the model.
- `maximum(Id, 1e-15)` is a hard clamp, active below about `Vg = -0.39 V`. A
  clamp is a C1 kink, and a Newton solve that lands on one stops converging.

The fix is to solve the series-resistance equation properly and take the
derivatives analytically. With `Id = f(Vg, Vd_int)` and
`Vd_int = Vd - Id*Rs`, the implicit function theorem gives

```
dId/dVd = f_d / (1 + f_d*Rs)
dId/dVg = f_g / (1 + f_d*Rs)
```

exact, cheaper than finite differences, and free of iteration noise. The
clamp is replaced by a smooth floor.

**This is a result in its own right**: the model returns an unconverged
current inside its own parameter bounds, and nothing in projects 12 or 13
caught it until a circuit needed derivatives. Report it that way, including
that stage 2 was checked and was unaffected.

## Job 1 — the solver

Modified nodal analysis with a damped Newton solve for DC, and backward-Euler
or trapezoidal integration for transient.

The compact model is **NMOS only**, so the circuits are NMOS-with-load. A CMOS
inverter is not available and claiming one would be wrong:

| circuit | analysis | metric |
|---|---|---|
| resistive-load inverter | DC sweep | switching threshold, small-signal gain |
| current mirror | DC | mirror ratio against output voltage |
| source follower | DC | output level, load sensitivity |
| ring oscillator | transient | oscillation frequency, stage delay |

The solver reports its own convergence: iteration counts, residual history,
and any node that failed. Those traces are the input to Job 3.

## Job 2 — do stage 2's corners still bound a real circuit?

Re-run stage 2's coverage-against-waste with circuit metrics in place of the
proxies.

**Prediction, registered before the measurement.** `worst_case_distance` was
built against `delay ∝ 1/Ion`. Real propagation delay depends on the whole
I-V trajectory the output swings through, not on the drive current alone, so
WCD corners are expected to **under-cover** real delay — the proxy's optimum
is not the circuit's optimum.

If that holds it is this stage's headline: a corner set optimised against a
proxy is not safe against the thing the proxy stood for. If it does not hold,
that is also reported, and it says the proxy was better than it looked.

The box corners are expected to keep covering, still wastefully. Stage 2
measured +50 to +81% waste against proxies; whether that figure survives a
real solve is the second number to watch.

## Job 3 — the agent diagnoses convergence failures

Stage 1 measured an LLM on a quantitative threshold judgement across three
prompt and payload conditions at 15 repeats. It never beat a rule and failed
identically every time. **That lowers the prior here and the prior is stated
before the measurement, not after.**

This stage tests a different shape of judgement: discrete classification from
structured evidence, with an answer key.

**Fault injection gives ground truth.** A fault of known type is injected into
the device model or the solve, the circuit is solved, and the trace is handed
to the diagnoser:

| injected fault | what it looks like |
|---|---|
| `c1_kink` | a clamp activated; residual stalls at a fixed point |
| `negative_gds` | wrong-sign output conductance; Newton diverges or oscillates |
| `truncated_inner_loop` | inner rs loop capped; derivative inconsistent with current |
| `bad_initial_guess` | converges from another start; no model defect |
| `singular_operating_point` | genuinely no solution at that bias |
| `none` | converged normally |

The diagnoser sees residual history, node voltages, iteration counts, which
device, and the bias — structured, the same for every arm. It returns one
class **or abstains**.

**Abstention is measured, not just permitted.** Accuracy alone rewards
confident guessing, and stage 1 already produced a condition with 100%
self-agreement that was systematically wrong. Reporting accuracy together
with abstention rate separates "knows the answer" from "is willing to say
something". A diagnoser that abstains on the cases it would have got wrong is
more useful than one with the same accuracy and no abstentions.

Rule baseline: heuristics over the residual pattern (stall vs divergence vs
oscillation, whether a clamp boundary was touched, whether the Jacobian
changed sign). Both arms see identical evidence; only the judgement differs.

## Scope and cut lines

- **3a** Job 0 + DC solver + inverter and mirror + corner re-validation. This
  is the core and stands alone.
- **3b** transient and the ring oscillator.
- **3c** the diagnostic agent with fault injection.

Each is releasable on its own.

## What this stage will not show

- A hand-written Newton solver is not SPICE. Its convergence behaviour is
  evidence about this model under this solver, and not about how a commercial
  simulator would treat it.
- NMOS-only circuits, because the model is an NMOS model.
- The model was never designed for circuit use. Finding that it needs work to
  survive one is the expected outcome, not an indictment.
- Fill the rest in from measurement. This section is required.
