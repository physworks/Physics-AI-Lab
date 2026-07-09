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

### 🔬 Project 1: TFT Id-Vg Modeling (GCA-based PINN)

The first hands-on implementation, using TFT drain-current behavior under the Gradual Channel Approximation as a controlled testbed for validating the PINN approach.

- [x] Formulate GCA governing equations (linear / saturation region)
- [x] Build synthetic Id-Vg dataset generator (clean + noisy)
- [x] PINN implementation (PyTorch)
- [x] Data-driven baseline for comparison
- [x] Fixed GCA boundary discontinuity (continuity correction) — PINN advantage narrowed from 14.9% to 5.2% once ground truth became smooth, a useful finding on when PINNs help most

→ [`projects/gca-pinn`](./projects/gca-pinn)

### 🔬 Project 2: Neural Operator — Vth Generalization (PhysicsNeMo FNO)

Extends Project 1 from point-wise PINN (fixed device) to operator learning across device parameters, using NVIDIA PhysicsNeMo's official FNO implementation.

- [x] Vth parameter-space dataset generation
- [x] FNO training via `physicsnemo.models.fno.FNO`
- [x] Generalization to held-out Vth values (not seen during training) — MSE 0.000052
- [ ] Extend to 2D (Vgs, Vds) operator
- [ ] Quantitative comparison vs. point-wise PINN

→ [`projects/neural-operator`](./projects/neural-operator)

### 🔬 Project 3: TCAD Surrogate — Self-built 1D Device Simulator

Sentaurus가 접근 불가능한 상황에서 직접 구현한 1D Poisson-Drift-Diffusion 소자 시뮬레이터. Box-integration(FVM)/FEM 이산화 + Newton-Raphson으로 MOS 커패시터의 비선형 Poisson 방정식을 풀고, quasi-static C-V 곡선으로 물리적 정확성을 검증.

- [x] 1D nonlinear Poisson solver (box-integration + Newton-Raphson)
- [x] C-V curve validation (accumulation-depletion-inversion)
- [x] Poisson-Drift-Diffusion coupling (Scharfetter-Gummel + Gummel iteration) — Shockley diode I-V law validated across 18 orders of magnitude
- [x] AI surrogate (FNO) trained on self-generated data — **657x speedup** (435ms → 0.66ms per curve), honest limitation noted in depletion-inversion transition region

→ [`projects/tcad-surrogate`](./projects/tcad-surrogate)

### Paper Reviews

- [x] (SK hynix + NVIDIA) Using AI Physics for Technology Computer-Aided Design Simulations
- [x] Physics-informed Neural Networks (PINNs)
- [x] DeepONet
- [x] Fourier Neural Operator
- [x] Physics-informed AI Accelerated Retention Analysis of Ferroelectric Vertical NAND
- [x] Revolutionizing TCAD Simulations with Universal Device Encoding and Graph Attention Networks
- [x] Physics-Informed Neural Network for Predicting Out-of-Training-Range TCAD Solution with Minimized Domain Expertise
- [x] FlashTP

### Other Planned Projects

- [ ] PINN for 1D Heat Equation
- [ ] PINN for Poisson Equation
- [ ] TCAD Surrogate Model (own FEM-based 1D device simulator, since Sentaurus is not currently accessible)

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
