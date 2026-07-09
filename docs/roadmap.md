# Roadmap

## Phase 1 — GCA-based PINN (TFT Id-Vg)
- [x] Formulate governing equations
- [x] Synthetic dataset generation
- [x] PINN implementation & training
- [x] Data-driven baseline comparison
- [x] Continuity-correction fix for GCA boundary discontinuity (PINN advantage: 14.9% → 5.2%, a useful finding on when physics constraints matter most)

## Phase 2 — Extension
- [x] Neural Operator (Vth generalization via PhysicsNeMo FNO) — held-out MSE 0.000052
- [x] TCAD-generated data validation (own FEM/box-integration 1D Poisson solver; Sentaurus not currently accessible) — C-V curve validated
- [x] Poisson-Drift-Diffusion coupling for full I-V (Scharfetter-Gummel + Gummel iteration, PN diode) — Shockley I-V law validated; solved ill-conditioning via shooting method
- [x] AI surrogate on self-generated TCAD data — 657x speedup (FNO, Na generalization), limitation noted in depletion-inversion region

## Phase 2 Complete — All planned milestones done

## Phase 3 — Consolidation
- [ ] Result write-up
- [ ] Paper review backlog (PINNs, FNO, DeepONet)
