# Physics-Informed Neural Networks (PINNs)

> Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019)
>
> Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.
>
> Journal of Computational Physics.

---

# Why I Read This Paper

Many semiconductor simulations are governed by physical equations such as

- Poisson Equation
- Drift-Diffusion Equation
- Heat Equation
- Diffusion Equation

Traditional numerical solvers (FDM/FEM/FVM) provide high accuracy but become computationally expensive for repeated simulations during design optimization.

PINNs provide an alternative by embedding physical laws directly into neural network training.

This paper is considered the foundation of Physics-Informed Machine Learning and is highly relevant to AI-driven semiconductor simulation.

---

# Core Idea

Instead of learning only from data,

PINNs learn from

```
Observed Data

+

Physical Laws (PDE)

↓

Neural Network
```

The network prediction is optimized to satisfy both.

---

# Loss Function

A PINN minimizes

```
Loss

=

Data Loss

+

Physics Loss
```

where

## Data Loss

Difference between prediction and measurements.

Typically

Mean Squared Error (MSE).

---

## Physics Loss

Residual of governing equations.

Example

Poisson Equation

```
∇²φ = -ρ / ε
```

The neural network output should satisfy this equation.

Automatic Differentiation computes derivatives directly from the network.

---

# Why This Is Important

Compared with purely data-driven models,

PINNs

✅ require less training data

✅ generalize better

✅ remain physically consistent

✅ improve extrapolation

---

# Limitations

The paper also discusses several challenges.

- Slow convergence
- Difficult optimization
- PDE balancing
- Scaling to high-dimensional problems

These limitations motivated later works such as

- Fourier Neural Operator
- DeepONet
- Neural Operators

---

# Relation to Semiconductor Simulation

Typical semiconductor simulations solve equations such as

- Poisson Equation
- Drift-Diffusion Equation
- Continuity Equation

PINNs provide a possible surrogate approach that approximates these physical systems while preserving governing equations.

Potential applications include

- TCAD acceleration
- Compact model generation
- Device parameter extraction
- Process optimization

---

# My Implementation Plan

Rather than directly applying PINNs to TCAD,

I will gradually increase complexity.

## Stage 1

Analytical GCA-based TFT Id-Vg

Objective

Understand PINN behavior in a controlled physical system.

---

## Stage 2

Poisson Equation

Understand PDE residual learning.

---

## Stage 3

Drift-Diffusion Equation

Toward semiconductor device physics.

---

## Stage 4

TCAD-generated Dataset

Physics AI for semiconductor device simulation.

---

# Key Takeaways

The biggest insight from this paper is that

> Physics should not merely generate training data.

Instead,

Physics itself becomes part of the optimization objective.

This idea fundamentally changes how machine learning can be applied to scientific and engineering problems.

---

# Personal Notes

This paper strongly aligns with my engineering philosophy.

Throughout my career, I have approached engineering problems by

Understand the Physics

↓

Model the System

↓

Deploy the Algorithm

PINNs extend the same philosophy by incorporating physical knowledge directly into AI training.

For this reason, this paper serves as the starting point of my Physics AI learning journey.
