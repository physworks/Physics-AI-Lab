# Physics-AI-Lab

> Understanding Physics.
> Building Algorithms.
> Accelerating Research with AI.

An open research portfolio focused on Physics-based Artificial Intelligence for semiconductor simulation and scientific computing.

This repository documents my journey from production algorithm development to Physics AI — including paper reviews, mathematical foundations, numerical methods, and hands-on implementations. It builds on 9+ years of experience developing physics-based compensation algorithms for OLED display panels (physical characterization → modeling → C++ production deployment at mass-production scale), now redirected toward semiconductor device simulation.

---

## Vision

Modern semiconductor R&D is evolving beyond conventional simulation.
Physics, numerical simulation, and AI are converging to accelerate scientific discovery.

The goal of this repository is to study, implement, and understand the technologies that will shape next-generation Virtual R&D.

---

## Research Areas

- Physics-informed Neural Networks (PINNs)
- Neural Operators
- Scientific Machine Learning
- Semiconductor Device Physics
- TCAD
- Numerical Methods
- Optimization Algorithms
- AI for Physical Systems

---

## Repository Structure

```
docs/            Research roadmap and references
paper-reviews/   Paper summaries and implementation notes
projects/        Implementation projects
notes/           Physics and mathematics notes
```

---

## Current Progress

### 🔬 Active Project: TFT Id-Vg Modeling (GCA-based PINN)

The first hands-on implementation, using TFT drain-current behavior under the Gradual Channel Approximation as a controlled testbed for validating the PINN approach.

- [x] Formulate GCA governing equations (linear / saturation region)
- [x] Build synthetic Id-Vg dataset generator (clean + noisy)
- [x] PINN implementation (PyTorch)
- [x] Data-driven baseline for comparison
- [x] Fixed GCA boundary discontinuity (continuity correction) — PINN advantage narrowed from 14.9% to 5.2% once ground truth became smooth, a useful finding on when PINNs help most
- [ ] Neural Operator extension

→ [`projects/gca-pinn`](./projects/gca-pinn)

### Paper Reviews

- [x] Physics-informed Neural Networks (PINNs)
- [ ] Fourier Neural Operator
- [x] DeepONet
- [ ] FlashTP

### Other Planned Projects

- [ ] PINN for 1D Heat Equation
- [ ] PINN for Poisson Equation
- [ ] TCAD Surrogate Model (Sentaurus-generated data)

---

## Long-term Goal

To build a practical portfolio in Physics AI that bridges

```
Physics
   ↓
Simulation
   ↓
Machine Learning
   ↓
Semiconductor Virtual R&D
```

---

## Disclaimer

This repository is an independent personal learning project.
The code and documents are implemented solely for educational and research purposes.