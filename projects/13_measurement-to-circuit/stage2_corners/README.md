# Stage 2 — corner and statistical models

How does device-to-device spread become a corner model, and do those corners
actually bound circuit behaviour?

```bash
pip install -r requirements.txt
python run_corners.py             # writes outputs/
python -m pytest tests/ -q        # 23 tests
```

Needs `12_compact-model-extraction` two levels up (or `CMEXT_PATH`), and
`stage1_design` beside this directory — stage 2 reuses stage 1's measurement
design and its Fisher analysis rather than reimplementing either.

## What was measured

**1 · Process correlation is not estimation correlation.** Both are
correlation matrices over the same seven parameters, and for `vth0`/`mu0` they
have **opposite signs**: process **−0.33**, estimation **+0.90**. Twelve of the
twenty-one pairs disagree in sign.

Stage 1 measured the estimation value — how the extraction trades two
parameters off against one noise realisation of one device. A corner model
needs the other quantity: how devices differ from each other. Reusing stage
1's number here would tilt every corner the wrong way, and it is an easy
mistake to make because the two objects look identical.

The process matrix is derived, not asserted. Six physical causes are sampled
(oxide thickness, channel length and width, interface traps, contact
resistance, work function) and the parameter correlations follow from what
each cause does to the device. The strongest coupling comes from the model
holding `COX` and `W/L` fixed: a thinner oxide raises the real `Cox`, and
since drain current goes as `mu·Cox`, the *fitted* `mu0` rises to absorb it
while `vth0` falls. Opposite directions, one cause.

**2 · The observed spread is not the process spread**, and the correction
works only where stage 1 said it would.

| | observed / true | after deconvolution |
|---|---|---|
| `vth0`, `ss`, `mu0`, `rs`, `eta` | 1.01 – 1.32× | **0.92 – 1.07×** |
| `theta` | 9.3× | 2.5× |
| `vsat` | 10.2× | 2.8× |

Subtracting the extraction covariance that stage 1 predicts recovers the true
spread within about 10% for the five parameters that are identifiable. It does
not rescue `theta` and `vsat` — the two stage 1 found to be weakly
identifiable or not identifiable at all. Almost none of their observed spread
is process variation; it is measurement noise, and a *linear* estimate of that
noise is no longer accurate when the noise is that large.

So a parameter's observed spread can be deconvolved only if the parameter was
measurable to begin with. For one that is not, the observed spread carries no
information about the process and no correction puts it back. Such a parameter
belongs fixed at nominal rather than given a range — the same decision stage 1
was measuring, reached from the other direction.

**3 · Coverage against waste.** A corner set that covers everything at twice
the necessary width is not a good corner set, and quoting only its coverage is
how over-wide corners survive review.

| method | corners | coverage | waste |
|---|---|---|---|
| `independent_box` | 128 | 100% | **+50 to +81%** |
| `one_at_a_time` | 14 | 95–98% — **fails** | −23 to −32% |
| `pca_corners` | 14 | 97–99% — **fails** | −6 to −27% |
| `statistical_mc` | 2000 | 99.4–99.9% | −4 to +14% |
| `worst_case_distance` | **2** | 99.7–99.9% | **−1 to 0%** |

The box covers everything because it reaches far outside the ellipsoid the
process actually occupies: a "3σ" box corner over seven parameters sits nearer
8σ in Mahalanobis distance. Targeting the ellipsoid instead bounds the same
population within about 0% waste using **2 simulations instead of 128**.

`one_at_a_time` and `pca_corners` both under-cover, for different reasons — a
sensitivity sweep is not a bound, and the metric's worst direction is not a
principal axis of the covariance.

## What this stage does not show

- The metrics are **proxies** computed from the compact model, not a circuit
  solve. `delay` ignores input slope, bias-dependent load, and every
  parasitic. Stage 3 replaces them and re-checks whether these corners still
  bound the real thing; if they stop bounding it, that is stage 3's finding.
- `worst_case_distance` needs the metric in advance, so it answers a narrower
  question than the others. It is not a general corner set, and it is scored
  only against the metric it was built for — scoring it across metrics it was
  not built for was an early mistake here and inverted the ranking.
- `statistical_mc` is excluded from the "narrowest safe" ranking. It is drawn
  from the same distribution the reference range comes from, so it reproduces
  that range by construction and would always appear to win, at the cost of
  hundreds of simulations.
- The sensitivity coefficients are first-order textbook relations, not values
  fitted to silicon. Everything is synthetic, with known ground truth.
- An earlier version omitted local per-parameter variation, which made two
  parameters sharing one cause come out at exactly −1.000 correlation. Real
  populations do not do that; local effects were added and the artefact is
  gone. The covariance is now full rank but strongly anisotropic — **3 axes
  carry 90% of the variance**, condition number 146 — which is the honest form
  of "the parameters are not seven independent dimensions".

## Layout

```
corners/
  device.py       resolves project 12's model and stage 1's analysis
  physical.py     physical causes -> parameter covariance; the sensitivity table
  deconvolve.py   observed spread minus the extraction covariance
  generate.py     the four corner methods
  metrics.py      proxy circuit metrics, batched
  evaluate.py     coverage against waste
```
