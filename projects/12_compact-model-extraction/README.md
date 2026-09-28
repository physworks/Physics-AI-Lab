# compact-model-extraction

Parameter extraction for a compact MOSFET model, built to answer a question a
low fitting residual cannot answer:

> **The curve matches. Can I trust the parameters?**

---

## Scope, stated first

This is **not** BSIM-CMG or any industry-standard compact model, and it does not
claim to be. It is a deliberately small analytic model with seven parameters,
chosen so that the **extraction procedure** and the **identifiability analysis**
are the subject rather than the model itself.

What it does contain: an EKV-style single-expression core valid across weak and
strong inversion, DIBL, vertical-field mobility degradation, velocity
saturation, and a self-consistently solved series resistance — enough that the
parameters interact the way real ones do.

What it does not contain: C-V, RF, noise, temperature, multiple geometries,
or any SPICE simulator coupling.

---

## The question

Most extraction demos end at "RMS residual 0.5%, good fit." That number says the
curve was reproduced. It says nothing about whether each parameter is
individually determined by the data.

Two parameters acting on the same measurement regime can trade off: many
different parameter sets give almost the same curve. The optimiser returns
whichever one it happened to land on, with no warning. A model card built from
those numbers will not extrapolate, and the failure shows up later, in circuit
simulation, where it is expensive.

This project measures that directly.

---

## Results

Synthetic device, known ground truth, 1% relative measurement noise, 15 noise
realisations. Everything below is reproduced by `python run_extraction.py`.

### The uncertainty analysis predicts the real spread

The Jacobian gives a predicted 1σ per parameter **without running any Monte
Carlo**. Against the measured spread over independent noise realisations:

| parameter | predicted 1σ | measured | sensitivity (Δlog₁₀Id per 1%) |
|---|---|---|---|
| `vth0` | 0.07% | 0.08% | 0.0302 |
| `ss` | 0.05% | 0.06% | 0.0186 |
| `mu0` | 0.58% | 0.65% | 0.0038 |
| `eta` | 0.47% | 0.52% | 0.0026 |
| `vsat` | 2.91% | 3.09% | 0.00048 |
| `rs` | 1.65% | 1.74% | 0.00037 |
| `theta` | 3.57% | 3.45% | 0.00035 |

Every parameter agrees within ~15%. That match is what makes the analysis a
usable pre-check rather than decoration: it tells you which parameters to trust
before you spend anything on repeated measurements.

Sensitivity spans **100×** across the seven parameters. `vth0` moves the curve
a hundred times more than `theta` does. No optimiser can recover from that —
it is a property of the measurement, not of the algorithm.

### What trades off against what

`vth0` ↔ `mu0`, **r = +0.925**.

Physically unsurprising: in the linear region Id ∝ μ(Vgs − Vth), so raising the
threshold and raising the mobility partly cancel. The consequence is concrete —
in subthreshold, Id ∝ μ₀·exp((Vgs−Vth)/nφt), so an error in μ₀ is absorbed by
`vth0` as a shift of n·φt·ln(μ_err). With μ₀ fixed at 2/3 of truth that predicts
−13.5 mV; the staged run shows −16 mV. The correlation is not an abstraction.

### Where each parameter stops being meaningful

Mean recovery error as measurement noise increases (8 seeds per level):

| parameter | 1% noise | 2% | 5% | 10% |
|---|---|---|---|---|
| `vth0` | 0.1% | 0.2% | 0.4% | 0.9% |
| `ss` | 0.1% | 0.1% | 0.2% | 0.4% |
| `eta` | 0.4% | 0.7% | 1.7% | 3.4% |
| `mu0` | 0.7% | 1.3% | 3.3% | 6.6% |
| `rs` | 1.0% | 2.1% | 5.3% | **11.0%** |
| `vsat` | 1.6% | 3.3% | 8.8% | **18.5%** |
| `theta` | 3.6% | 6.9% | **16.8%** | **32.7%** |

`theta` crosses 10% error somewhere between 2% and 5% measurement noise; `vsat`
and `rs` hold until 10%. So below ~2% noise `theta` is a number; above it,
`theta` is the optimiser's opinion — and the fit reports both identically.

---

## The finding worth reading

Two extraction strategies were compared, and **the interesting result is that
the one with the worse recovery is behaving correctly.**

**Staged extraction** (subthreshold → DIBL → linear → saturation, gated after
each stage) escalates on **15/15** runs. The linear stage drives `theta` to its
lower bound, the gate catches it as *pinned at bound — the data did not
determine it*, and the run stops rather than hand over the value.

**Simultaneous extraction** (all seven at once) fits well — 0.1–3.7% recovery at
1% noise — and escalates **0/15**. It returns `theta = 0.347` with no caveat.

At 1% noise the simultaneous answer is genuinely better. At 5% noise the same
code returns a `theta` that is 17% wrong, and at 10% noise, 33% wrong — with the
same confident silence. The staged run's escalation was never a failure to
extract; it was the only one of the two that reported what the data could not
determine.

Neither approach is "right." The identifiability analysis is what lets you
choose, and it is the part most extraction workflows skip.

A correlation-driven refinement is also included: after the staged run, the
parameters the correlation matrix flags as exchangeable are released and refitted
*jointly*, since individually they are exactly the ones the data cannot separate.
This cuts mean recovery error from ~34% to ~20% — real improvement, and still
not enough to make `theta` trustworthy, because no fitting strategy can recover
a parameter the measurement does not constrain.

---

## Running it

```bash
pip install numpy scipy matplotlib
python run_extraction.py            # full run -> outputs/
python run_extraction.py --quick    # fast check
python tests/test_cmext.py          # 14 tests, no pytest needed
```

Outputs land in `outputs/`: `report.md`, `results.json`, and two figures.

The test suite checks the invariants rather than the numbers: that the generated
data's subthreshold swing equals the parameter that set it, that the DIBL shift
equals η·ΔVds, that a sub-thermal swing fails its gate, that a parameter pinned
at a bound is flagged, and that the predicted uncertainty matches Monte Carlo.

---

## Layout

```
cmext/
  model.py            7-parameter EKV-style compact model + synthetic measurement
  extraction.py       staged extraction, physics gates, correlation refinement
  identifiability.py  sensitivity, Jacobian covariance, correlation, noise sweep
run_extraction.py     runs everything, writes the report and figures
tests/test_cmext.py   14 tests
```

---

## Limitations

- Single device geometry. Separating `theta` from `rs` properly needs several
  channel lengths — `rs` is geometry-independent, mobility degradation is not.
  With one geometry the ambiguity is real and is reported rather than hidden.
- The data is generated by the same model that is fitted, so there is no model
  mismatch. Real extraction also fights structural error in the model itself,
  which this cannot show.
- Seven parameters, not the ~100 of a production compact model. The staging and
  identifiability ideas carry over; the bookkeeping does not.
- No C-V, RF, noise, or temperature. Those regimes determine parameters this
  model does not have.
