# DeepONet

> **Lu et al. (2021)**
>
> **Learning Nonlinear Operators via DeepONet Based on the Universal Approximation Theorem of Operators**
>
> **Nature Machine Intelligence**

---

## Why I Read This Paper

DeepONet is one of the foundational architectures in **Scientific Machine Learning (SciML)** for learning nonlinear operators. Unlike conventional neural networks that learn mappings between finite-dimensional vectors, DeepONet learns mappings **between functions**, making it highly suitable for solving PDE-based problems such as semiconductor device simulation.

I read this paper to understand how Operator Learning differs from Physics-Informed Neural Networks (PINNs) and how it can be applied to accelerate **TCAD simulations** and AI-based surrogate models.

---

## Problem

Traditional neural networks learn a mapping

\[
x \rightarrow y
\]

where both input and output are finite-dimensional vectors.

However, many scientific and engineering problems require learning an **operator**, which maps one function to another:

\[
\mathcal{G}: u(x) \rightarrow s(x)
\]

Examples include:

- Solving partial differential equations (PDEs)
- Fluid dynamics
- Heat transfer
- Electromagnetic field simulation
- Semiconductor device simulation (TCAD)

Training an independent neural network for every boundary condition or material configuration is computationally expensive.

The challenge is to learn a **general nonlinear operator** that can predict solutions for many different PDE instances after a single training process.

---

## Key Idea

DeepONet introduces the concept of **Operator Learning**.

Instead of approximating a single solution, it learns the operator itself.

```
Boundary Condition / Material Property
                │
                ▼
            DeepONet
                │
                ▼
         Solution Function
```

Once trained, the same network can predict solutions for unseen boundary conditions and device parameters without repeatedly solving numerical PDEs.

---

## Method

DeepONet consists of two neural networks.

### 1. Branch Network

The Branch Network receives sampled values of the input function.

Examples include:

- Boundary conditions
- Initial conditions
- Material properties
- Device geometries

It outputs latent feature vectors:

\[
b_1,b_2,\cdots,b_p
\]

---

### 2. Trunk Network

The Trunk Network receives spatial coordinates:

\[
x
\]

It outputs basis functions:

\[
t_1(x),t_2(x),...,t_p(x)
\]

---

### Final Prediction

The prediction is computed as the inner product of the Branch and Trunk outputs:

\[
G(u)(x)
=
\sum_{i=1}^{p}
b_i(u)t_i(x)
\]

This enables DeepONet to approximate nonlinear operators rather than individual function values.

---

## Equation

DeepONet approximates the nonlinear operator

\[
G:u\rightarrow s
\]

using

\[
G(u)(x)
=
\sum_{k=1}^{p}
b_k(u)t_k(x)
\]

where

- \(b_k\): Output of the Branch Network
- \(t_k(x)\): Output of the Trunk Network

Unlike standard neural networks, DeepONet learns mappings between functions instead of mappings between vectors.

---

## Advantages

- Learns nonlinear operators instead of individual PDE solutions.
- Generalizes to unseen boundary conditions.
- Much faster inference than repeatedly solving numerical PDEs.
- Applicable to a wide range of scientific computing problems.
- Strong theoretical foundation based on the Universal Approximation Theorem for Operators.
- Well suited for surrogate modeling in computational physics.

---

## Limitations

- Requires a large amount of high-quality simulation data.
- Performance depends on the diversity of training data.
- Does not explicitly enforce physical laws like PINNs.
- Accuracy may degrade outside the training distribution.
- Training cost can be relatively high.

---

## Key Takeaways

- DeepONet is one of the first practical frameworks for **Operator Learning**.
- It predicts entire functions rather than scalar values.
- Unlike PINNs, it primarily relies on supervised simulation data instead of physics-based loss functions.
- DeepONet is highly promising for **AI-accelerated TCAD surrogate models**, where repeated semiconductor device simulations under varying process conditions are required.
- It established the foundation for later Neural Operator architectures such as the Fourier Neural Operator (FNO).

---

## Personal Notes

### Relation to My Current Project

My current GitHub project implements a PINN for TFT Id–Vg characteristics based on the Gradual Channel Approximation (GCA).

DeepONet provides another promising approach by learning the solution operator directly rather than enforcing physical equations during optimization.

A natural next step is to compare PINN and DeepONet on the same TFT dataset and evaluate their generalization performance under unseen device parameters such as threshold voltage, mobility, oxide capacitance, and channel dimensions.

### Relation to TCAD

From a semiconductor TCAD perspective, DeepONet has strong potential as an AI-based surrogate model.

Instead of repeatedly running computationally expensive device simulations, a trained DeepONet could directly predict electrical characteristics under different material properties and process conditions, significantly reducing simulation time during Virtual R&D.

### Implementation Plan

- [ ] Implement DeepONet using PyTorch
- [ ] Train on synthetic TFT GCA dataset
- [ ] Compare DeepONet with PINN
- [ ] Compare with a pure MLP baseline
- [ ] Evaluate generalization on unseen device parameters
- [ ] Extend to TCAD surrogate modeling

---

### Paper Link

https://www.nature.com/articles/s42256-021-00302-5
