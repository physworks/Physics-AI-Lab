# Stage 2 — corner and statistical models

**Question.** How does device-to-device spread become a corner model, and do
those corners actually bound circuit behaviour?

Input is stage 1's output: the selected measurement design and the extraction
procedure. Output is a set of corner models plus the evidence for whether they
are usable.

Two decisions are already settled:

- **No LLM arm in this stage.** Stage 1 measured the agent on a quantitative
  threshold judgement across three prompt and payload conditions at 15 repeats
  and it never beat a rule, failing identically every time. Running the same
  shape of experiment again would not be informative. The agent moves to stage
  3, where the judgement is discrete and structural rather than a threshold.
- **Correlation comes from physical common causes**, derived, not asserted as
  a covariance matrix.

## 1. The generative process model

A covariance matrix written down by hand encodes the answer. Instead, sample
*physical* variables and let the parameter correlations fall out of the device
physics:

| physical variable | spread | parameters it moves |
|---|---|---|
| oxide thickness `tox` | ~2% | `vth0` (via Cox), `mu0` (vertical field), `theta` |
| channel length `L` | ~3% | drive current, `vsat`, DIBL (`eta`) |
| channel width `W` | ~2% | drive current |
| interface trap density `Dit` | ~10% | `ss`, `vth0` shift |
| contact / series resistance | ~8% | `rs` |
| flatband / work function | ~1.5% | `vth0` |

Each device draws these independently, and the compact-model parameters are
computed from them. The resulting parameter covariance is then a *consequence*
of shared physical causes, which is what makes `vth0`–`mu0` correlation real
rather than stipulated.

Keep the physical-to-parameter map in one module with the relationships
written out explicitly. It is the part a reviewer will check first.

> **Do not reuse stage 1's measured `vth0`–`mu0` correlation of 0.925 here.**
> That is an *estimation* correlation — how the extraction trades the two off
> against one noise realisation. Process correlation is a different quantity
> with a different cause. Conflating them is the most likely way to get this
> stage wrong, and showing the two side by side is itself a result worth
> reporting.

## 2. The contamination problem

What is observed is not the process spread:

```
observed spread  =  process spread  ⊕  extraction uncertainty
```

Build corners from the observed spread and they over-bound, because part of
the width is measurement noise rather than device variation. The extraction
uncertainty is exactly the quantity stage 1 minimised, so the two stages meet
here.

Report both: corners from raw observed spread, and corners after subtracting
the extraction covariance that stage 1's Fisher analysis predicts. Quantify
how much margin the naive version wastes. Verify the subtraction against the
known truth — the ground-truth process spread is available because the
population was generated.

## 3. Corner generation methods

| method | what it does |
|---|---|
| `independent_sigma` | ±3σ per parameter, one at a time. Common, and ignores correlation |
| `pca_corners` | corners along the principal axes of the covariance |
| `statistical_mc` | sample the fitted distribution directly |
| `worst_case_distance` | search the kσ ellipsoid for the point that worst-cases a given metric |

## 4. Evaluation — coverage against waste

Neither number alone decides anything, and the trade between them is the
result.

- **Coverage**: of N devices drawn from the true process distribution, the
  fraction whose metric falls inside the range the corners predict. Target is
  the nominal (99.7% for ±3σ); below it the corners are unsafe.
- **Waste**: how much wider the corner-predicted range is than the true range.
  This is design margin spent on nothing.

Expect `independent_sigma` to have high coverage and large waste. How large is
the number this stage produces.

## 5. The circuit metric is a proxy here

There is no circuit solver until stage 3. Use metrics computable from the
compact model alone:

- drain current at specified operating points
- a delay proxy, `C·V/I` at a specified swing
- an on/off ratio

Stage 3 replaces the proxy with an actual Newton-Raphson circuit solve and
re-checks whether these corners still bound it. **If they stop bounding it,
that is stage 3's finding, and this stage should be written so that outcome is
visible rather than embarrassing.** Say in the report that the proxy is a
proxy.

## 6. Tests worth writing

- The physical-to-parameter map reproduces a known `vth0`–`mu0` correlation
  sign and rough magnitude from `tox` variation alone.
- Noise deconvolution recovers the known process spread when the extraction
  covariance is known, within a stated tolerance.
- Corners generated from a correlated population bound a larger fraction of
  devices than independent corners at matched width (or state plainly that
  they do not).
- Coverage of `statistical_mc` approaches its nominal level as N grows.
- Estimation correlation and process correlation are computed by separate
  functions and are not interchangeable.

## 7. What this stage will not show

Fill this in from measurement, not from expectation. It is a required section.
