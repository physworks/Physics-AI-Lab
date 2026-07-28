# Paper Name

> Ilyes Batatia et al. (2022)
>
> **MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields**
>
> NeurIPS 2022

---

# Why I Read This Paper

NequIP은 E(3)-Equivariant GNN을 통해 매우 높은 정확도를 달성했고,

Allegro는 이를 산업에서 사용할 수 있도록 빠르게 만들었다.

그런데 한 가지 문제가 남아 있었다.

> "Message Passing을 여러 번 해야 Many-body Interaction을 표현할 수 있을까?"

MACE는

**Higher-order Message Passing**이라는 새로운 아이디어를 통해

- 더 적은 Layer
- 더 높은 정확도
- 더 빠른 계산

을 동시에 달성하였다.

현재 MLIP 분야에서 가장 널리 사용되는 모델 중 하나이며,

Google DeepMind, Microsoft, Meta, Cambridge 등에서도 활발히 활용되고 있다. :contentReference[oaicite:0]{index=0}

---

# Background

기존 MLIP의 발전은 다음과 같다.

```text
DeepPot-SE
      │
      ▼
NequIP
      │
      ▼
Allegro
      │
      ▼
MACE
```

NequIP의 Message Passing은

기본적으로

**Two-body Interaction**을 반복해서 전달한다.

즉

```
Atom A

↓

Atom B

↓

Atom C
```

처럼 여러 Layer를 거쳐야

복잡한 Many-body 효과를 표현할 수 있다.

---

# Core Idea

MACE의 핵심 아이디어는 매우 직관적이다.

> **한 번의 Message 안에 Higher-order Interaction을 넣자.**

즉

기존에는

```
2-body

↓

2-body

↓

2-body
```

를 여러 번 수행했다면,

MACE는

```
4-body

↓

Message Passing

↓

Energy
```

처럼

한 번의 Message만으로도

복잡한 상호작용을 표현한다. :contentReference[oaicite:1]{index=1}

---

# Higher-order Message Passing

기존 Equivariant GNN

```
Atom

↓

Neighbor

↓

Message

↓

Update
```

MACE

```
Atom

↓

Neighbor Set

↓

Higher-order Correlation

↓

Message

↓

Update
```

즉

Neighbor 하나가 아니라

여러 Neighbor의 관계를

동시에 학습한다.

---

# Body Order

Body Order란

몇 개의 원자가 동시에 상호작용하는지를 의미한다.

```
2-body

A — B

------------

3-body

A — B
 \ /
  C

------------

4-body

A — B
|\ /|
| X |
|/ \|
C — D
```

MACE는

Higher-order Correlation을 직접 표현하므로

깊은 네트워크가 필요하지 않다. :contentReference[oaicite:2]{index=2}

---

# Architecture

전체 구조는

```
Atomic Coordinates

↓

Neighbor Graph

↓

Equivariant Message

↓

Symmetric Higher-order Product

↓

Atomic Feature

↓

Energy

↓

Force
```

이다.

---

# Atomic Cluster Expansion (ACE)

MACE의 이름은

**Atomic Cluster Expansion**에서 유래하였다.

ACE는

원자 주변 환경을

Many-body Basis Function으로 표현하는 물리 기반 방법이다.

MACE는

이를

Neural Network와 결합하였다.

즉

```
ACE

+

Equivariant GNN

↓

MACE
```

이다. :contentReference[oaicite:3]{index=3}

---

# Why Faster?

NequIP은

Many-body Interaction을 만들기 위해

여러 Layer가 필요하다.

반면

MACE는

Higher-order Message를 사용하기 때문에

2개 정도의 Layer만으로도

동일하거나 더 높은 표현력을 얻는다.

즉

```
NequIP

Layer

↓

Layer

↓

Layer

↓

Many-body

----------------

MACE

Higher-order Message

↓

Many-body
```

가 가능하다. :contentReference[oaicite:4]{index=4}

---

# Performance

논문에서는

다음 Benchmark에서 평가하였다.

- rMD17
- 3BPA
- AcAc

결과는

- 당시 SOTA 수준의 Force Prediction
- 더 적은 Layer
- 빠른 학습
- 빠른 추론

을 동시에 달성하였다.

특히 적은 학습 데이터에서도 더 좋은 학습 곡선(Data Efficiency)을 보였다. :contentReference[oaicite:5]{index=5}

---

# Why This Paper Matters

MACE 이후

Foundation Model까지

많은 MLIP가

Higher-order Representation을 사용하게 되었다.

대표적으로

- MACE-MP
- MACE-OFF
- MatterSim
- MACE-H

등이 등장하였다. :contentReference[oaicite:6]{index=6}

---

# Semiconductor Relevance

반도체 Material Simulation에서는

다음과 같은 문제가 중요하다.

- Defect
- Vacancy
- Dopant Diffusion
- Grain Boundary
- Oxide Interface
- Stress
- Thermal Annealing

이들은

단순한 Pair Interaction만으로는

충분히 표현되지 않는다.

Higher-order Interaction을 사용하는

MACE는

이러한 복잡한 환경을

더 정확하게 모델링할 수 있다.

---

# Relation with SK hynix TCAD

Material Simulation 관점에서는

```text
DFT

↓

MACE

↓

Large-scale Molecular Dynamics

↓

Material Property

↓

TCAD Parameter

↓

Device Simulation
```

이라는

Physics AI Pipeline을 구축할 수 있다.

향후

Physics-informed TCAD,

AI-assisted Process Simulation,

Surrogate Modeling과도

직접 연결된다.

---

# Limitations

## 1. 계산 복잡도

Higher-order Correlation을 사용하므로

구현이 복잡하다.

---

## 2. 메모리 사용량

Higher-order Tensor 계산 때문에

GPU 메모리 사용량이 증가한다.

---

## 3. DFT Dataset 필요

다른 MLIP와 마찬가지로

고품질 DFT 데이터가 필요하다.

---

## 4. Long-range Interaction

장거리 Coulomb Interaction은

별도의 모델과 결합하는 경우가 많다.

---

# NequIP vs Allegro vs MACE

| 항목 | NequIP | Allegro | MACE |
|------|---------|----------|---------|
| Equivariant | O | O | O |
| Message Passing | O | X | O |
| Higher-order Correlation | 간접 | 제한적 | 직접 |
| 데이터 효율 | 매우 높음 | 높음 | 매우 높음 |
| 추론 속도 | 보통 | 매우 빠름 | 빠름 |
| 정확도 | 매우 높음 | 매우 높음 | 최고 수준(SOTA) |

---

# My Insight

MACE를 공부하면서 가장 인상 깊었던 점은

**"깊은 네트워크보다 더 풍부한 물리적 상호작용을 표현하는 것이 중요하다."**는 것이다.

NequIP은 Layer를 쌓아 Many-body 효과를 만들어냈다면,

MACE는 **Higher-order Correlation 자체를 Message 안에 포함**시켰다.

즉,

모델을 단순히 더 깊게 만드는 것이 아니라

**물리적으로 의미 있는 표현을 더 풍부하게 만드는 방향**으로 발전한 것이다.

Physics AI의 관점에서 보면

MACE는 현재 MLIP 분야의 핵심 모델이며,

향후 Semiconductor TCAD에서도

Material Simulation을 AI로 대체하는 중요한 기반 기술이 될 가능성이 매우 높다고 생각된다.

---

# Keywords

- MACE
- Atomic Cluster Expansion
- Higher-order Message Passing
- Equivariant GNN
- Machine Learning Interatomic Potential
- MLIP
- Molecular Dynamics
- Many-body Interaction
- Materials Informatics
- Physics AI