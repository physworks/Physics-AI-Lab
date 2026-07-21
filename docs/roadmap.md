# Roadmap

## Phase 1 — GCA-based PINN (TFT Id-Vg)
- [x] Formulate governing equations
- [x] Synthetic dataset generation
- [x] PINN implementation & training
- [x] Data-driven baseline comparison
- [x] Continuity-correction fix for GCA boundary discontinuity (PINN advantage: 14.9% → 5.2%, a useful finding on when physics constraints matter most)

## Phase 2 — Neural Operator & TCAD Surrogate
- [x] Neural Operator (Vth generalization via PhysicsNeMo FNO) — held-out MSE 0.000052
- [x] Self-built 1D Poisson solver (box-integration/FEM + Newton-Raphson) — C-V curve validated
- [x] Poisson-Drift-Diffusion coupling for full I-V (Scharfetter-Gummel + Gummel iteration, PN diode) — Shockley I-V law validated across 18 orders of magnitude; solved ill-conditioning via shooting method
- [x] AI surrogate on self-generated TCAD data — 657x speedup (FNO, Na generalization)

## Phase 3 — Material Simulation & Quantum Transport
- [x] Toy MLIP (Lennard-Jones MD + Behler-Parrinello style NNP) — energy generalization validated, force-matching limitation documented
- [x] NEGF quantum transport (1D tight-binding + semi-infinite leads) — uniform chain T(E)=1 validated, double-barrier resonance, Landauer I-V with NDR

## Phase 4 — Consolidation & Extension
- [x] Paper review backlog — 16 papers reviewed (see `paper-reviews/`)
- [ ] PINN for 1D Heat / Poisson Equation
- [ ] NEGF I-V curve noise reduction (finer (E,V) grid or adaptive integration)
- [ ] MLIP force-matching accuracy improvement (angular symmetry functions)
- [ ] ML-accelerated NEGF (after AD-NEGF, DeePTB-NEGF reviews)
