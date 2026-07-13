# Physics-Informed Graph Neural Networks

> Johannes Brandstetter, Daniel Worrall, Max Welling (2022)
>
> **Message Passing Neural PDE Solvers**
>
> International Conference on Learning Representations (ICLR 2022)

---

## Why I Read This Paper

MeshGraphNet은 Graph Neural Network(GNN)를 이용해 물리 시뮬레이션을 학습할 수 있음을 보여주었지만, 기본적으로는 **Data-driven Learning**에 가깝습니다.

실제 반도체 TCAD에서는 단순히 데이터를 잘 맞추는 것보다 **Poisson Equation, Drift-Diffusion Equation, Heat Equation 등 물리 법칙을 항상 만족하는 것**이 훨씬 중요합니다.

이 논문은 Graph Neural Network에 Physics Constraint를 결합하여 **Physics-Informed Graph Neural Network(PIGNN)**의 방향을 제시한 대표적인 연구입니다.

특히 향후 AI 기반 TCAD Surrogate Model에서는

- MeshGraphNet
- PINN
- Graph Neural Network

가 자연스럽게 결합될 가능성이 매우 높기 때문에 반드시 이해해야 하는 논문입니다.

---

## Problem

MeshGraphNet은

```
Mesh

↓

Graph Neural Network

↓

Next State
```

를 학습합니다.

하지만

Physics Constraint가 없기 때문에

- 데이터가 부족하면 성능 저하
- Long Rollout에서 Error Accumulation
- Energy Conservation 위반
- Boundary Condition 위반

등이 발생할 수 있습니다.

즉,

```
Simulation처럼 보이는 Prediction

≠

Physics-consistent Prediction
```

이라는 문제가 존재합니다.

---

## Key Idea

논문의 핵심은

**Graph Message Passing 과정에서 Physics를 직접 학습시키는 것**입니다.

기존 GNN

```
Graph

↓

Message Passing

↓

Prediction
```

제안 방식

```
Graph

↓

Message Passing

↓

Physics Residual

↓

Prediction
```

즉,

Prediction Accuracy뿐 아니라

Physics Equation Residual까지 동시에 최소화합니다.

이는

PINN의 Physics Loss를

Graph Neural Network에 적용한 개념으로 이해할 수 있습니다.

---

## Method

### Step 1. Mesh → Graph

Simulation Mesh를

Graph로 변환합니다.

Node

- Position
- Velocity
- Pressure
- Temperature
- Material Property

Edge

- Neighbor Information
- Relative Distance
- Connectivity

를 저장합니다.

---

### Step 2. Message Passing

각 Node는

인접 Node와 정보를 교환합니다.

```
Node

↓

Neighbor Message

↓

Aggregation

↓

Node Update
```

이를 여러 Layer 반복합니다.

---

### Step 3. Physics Residual 계산

Prediction 이후

PDE Residual을 계산합니다.

예를 들어

Poisson Equation

\[
\nabla^2\phi+\rho/\varepsilon=0
\]

라면

Residual

\[
R(x)
=
\nabla^2\phi
+
\rho/\varepsilon
\]

를 계산합니다.

Residual이

0에 가까울수록

Physics를 잘 만족합니다.

---

### Step 4. Joint Loss

Loss는

```
Prediction Loss

+

Physics Loss
```

를 동시에 최소화합니다.

---

## Equation

Graph는

\[
G=(V,E)
\]

Message Passing은

\[
m_{ij}
=
\phi_e(h_i,h_j,e_{ij})
\]

Node Update

\[
h_i'
=
\phi_v
\left(
h_i,
\sum_jm_{ij}
\right)
\]

Physics Loss

\[
L_{physics}
=
\|R(x)\|^2
\]

최종 Loss

\[
L
=
L_{data}
+
\lambda
L_{physics}
\]

입니다.

여기서

- \(L_{data}\) : 실제 Simulation과의 오차
- \(L_{physics}\) : PDE Residual
- \(\lambda\) : Physics Constraint의 중요도

입니다.

---

## Advantages

- 데이터가 적어도 높은 정확도를 유지한다.
- Physics Consistency를 유지한다.
- Long Rollout 안정성이 향상된다.
- Mesh 구조를 그대로 활용할 수 있다.
- 다양한 PDE에 적용 가능하다.
- TCAD와 같은 Multi-Physics 문제에 적합하다.

---

## Limitations

- Physics Residual 계산 비용이 추가된다.
- 복잡한 PDE에서는 Residual 계산이 어렵다.
- Loss Weight(λ) 설정에 민감하다.
- Multi-Physics Coupling에서는 구현 난이도가 높다.
- 대규모 Graph에서는 GPU 메모리 요구량이 증가한다.

---

## Key Takeaways

- MeshGraphNet은 데이터를 학습한다.
- Physics-Informed GNN은 물리 법칙까지 함께 학습한다.
- PINN과 GNN을 결합한 형태로 이해할 수 있다.
- Physics Constraint 덕분에 일반화 성능과 안정성이 향상된다.
- 차세대 AI 기반 TCAD에서 매우 유망한 접근 방식이다.

---

## Personal Notes

### Relation to My Current Project

현재 진행 중인 GitHub 프로젝트는

```
GCA Equation

↓

PINN

↓

Id-Vg Prediction
```

이다.

향후에는

```
TCAD Mesh

↓

Graph Neural Network

↓

Physics Residual

↓

Device Prediction
```

으로 확장해 보고 싶다.

즉,

현재의 PINN 프로젝트를

Graph 기반으로 발전시키는 다음 단계가 될 수 있다.

---

### Relation to My Career

Physics-Informed GNN 역시

```
Physics

↓

Graph Representation

↓

AI Learning

↓

Simulation
```

이라는 동일한 철학을 가진다.

---

### Relation to SK hynix TCAD

SK하이닉스가 추진하는

- AI Physics
- Virtual R&D
- TCAD Acceleration
- Material Simulation

을 구현하기 위한 매우 중요한 기술이다.

특히

```
TCAD Mesh

↓

Graph Neural Network

↓

Physics Constraint

↓

Surrogate Model
```

은 향후 차세대 Device Simulation의 핵심 구조가 될 가능성이 높다.

---

### Comparison with Previous Papers

| Paper | 입력 형태 | Physics 적용 | 특징 |
|---------|------------|--------------|------|
| PINN | Point | 매우 높음 | PDE Residual 기반 |
| DeepONet | Function | 높음 | Operator Learning |
| FNO | Grid | 높음 | Fourier Domain Operator |
| MeshGraphNet | Graph | 중간 | Mesh 기반 Simulation |
| Physics-Informed GNN | Graph | 매우 높음 | Graph + Physics Constraint |

Physics-Informed GNN은

MeshGraphNet에 PINN의 Physics Loss를 결합한 형태로 이해할 수 있다.

---

### Relation to TCAD

TCAD에서는

- Device Mesh
- Poisson Equation
- Drift-Diffusion Equation
- Heat Equation
- Quantum Correction

등 여러 PDE를 동시에 계산한다.

Physics-Informed GNN은

이러한 Multi-Physics 문제를

Graph 구조 위에서 효율적으로 학습할 수 있는 유망한 접근 방식이다.

향후

AI 기반 TCAD Solver의 핵심 기술이 될 가능성이 높다.

---

### Future Learning Plan

이 논문를 기반으로 다음 프로젝트를 진행할 계획이다.

- [ ] PyTorch Geometric 학습
- [ ] Graph Message Passing 구현
- [ ] Physics Loss 추가
- [ ] MeshGraphNet + Physics Constraint 구현
- [ ] TCAD Mesh 데이터셋 적용
- [ ] TCAD Surrogate Model 구축

---

### Paper Link

https://arxiv.org/abs/2202.03376