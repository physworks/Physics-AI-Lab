# GCA-based TFT Id-Vg PINN

[← Back to Physics-AI-Lab](../../README.md)

Applying **Physics-Informed Machine Learning** to TFT device characteristics, as the first hands-on project in [Physics-AI-Lab](../../README.md).

This project bridges two things I've spent my career on: physical system modeling and production-grade algorithm deployment — now applied to the kind of problems TCAD / Physics AI research tackles (accelerating device simulation with AI while preserving physical validity).

---

## Why this project

Classical device simulation (TCAD, SPICE-level compact models) is physically accurate but computationally expensive, especially when scanning wide parameter spaces for process/design optimization. Pure data-driven ML models are fast but can violate physical laws outside their training distribution.

**Physics-Informed Neural Networks (PINNs)** address this by embedding the governing physical equations directly into the training loss, so the model is constrained to stay physically consistent even in sparse-data regions.

This project starts with a well-understood, analytically tractable case — **TFT (Thin-Film Transistor) Id-Vg characteristics under the Gradual Channel Approximation (GCA)** — as a controlled environment to validate the PINN approach before extending to more complex device physics.

---

## Project 1: GCA-based TFT Id-Vg PINN

### Physics background

TFT drain current follows GCA, split into two operating regions:

**Linear region** (Vds < Vgs − Vth):

```
Id = μ · Cox · (W/L) · [(Vgs − Vth)·Vds − Vds²/2]
```

**Saturation region** (Vds ≥ Vgs − Vth):

```
Id = 0.5 · μ · Cox · (W/L) · (Vgs − Vth)² · (1 + λ·Vds)
```

Real TFTs deviate from this ideal model due to subthreshold behavior, mobility degradation, and interface trap effects — which is exactly where a data-informed correction on top of the physical baseline becomes useful, and where this project is headed next (see Roadmap).

### Approach

1. Generate synthetic Id-Vg data from the analytical GCA model (with and without measurement noise) as a controlled ground truth
2. Train a PINN where the loss combines:
   - **Data loss**: MSE against (noisy) synthetic measurements
   - **Physics loss**: residual of the GCA equation evaluated at collocation points
3. Compare against a pure data-driven baseline (same architecture, no physics loss) to quantify what the physics constraint actually buys — particularly in low-data and noisy regimes

### Status

| Step | Status |
|---|---|
| GCA synthetic data generator | ✅ Done |
| PINN model implementation (PyTorch) | 🔄 In progress |
| Data-driven baseline for comparison | ⬜ Planned |
| Neural Operator extension (structure generalization) | ⬜ Planned |
| Write-up with results | ⬜ Planned |

---

## Repo structure

```
Physics-AI-Lab/
├── data/               # Generated datasets & device parameter configs
├── src/                # Data generation, model, training scripts
├── models/             # Saved model checkpoints
├── notebooks/          # Exploration & result visualization
└── README.md
```

---

## Roadmap

- [x] GCA-based Id-Vg synthetic dataset
- [ ] PINN implementation & validation
- [ ] Data-driven baseline comparison
- [ ] Neural Operator extension
- [ ] TCAD surrogate model experiment (Sentaurus-generated data)
- [ ] Write-up / portfolio consolidation

---

## Related

Part of [Physics-AI-Lab](../../README.md) — see the main README for background and the broader research roadmap.
