# Paper Name

> Simon Batzner et al. (2021)
>
> **E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials (NequIP)**
>
> Nature Communications (2022), arXiv 2021

---

# Why I Read This Paper

DeepPot-SE까지는 원자 주변 환경을 **Invariant Descriptor**로 변환한 뒤 Neural Network에 입력하였다.

하지만 여기서 한 가지 의문이 생긴다.

> "회전 정보를 굳이 버려야 할까?"

NequIP은 이 질문에서 출발한다.

회전 정보를 제거(Invariant)하는 대신,

**회전에 따라 Feature도 함께 회전하도록(Equivariant)** 학습하면 더 많은 물리 정보를 보존할 수 있다는 것이 핵심 아이디어이다.

NequIP은 이후

- Allegro
- MACE
- SevenNet
- Orb
- MatterSim

등 최신 MLIP들의 기반이 되었다. :contentReference[oaicite:0]{index=0}

---

# Background

기존 MLIP의 발전 과정은 다음과 같다.

```
Classical Potential

↓

DeepMD

↓

DeepPot-SE

↓

NequIP
```

DeepPot-SE는

- Rotation Invariant
- Translation Invariant
- Permutation Invariant

을 만족하였다.

하지만

원자 간 방향(Direction)에 대한 정보는 상당 부분 잃게 된다.

NequIP은

방향 자체를 학습한다.

---

# Core Idea

핵심 아이디어는

> **Geometry를 없애지 말고 그대로 학습하자.**

즉

```
Atomic Coordinates

↓

Graph

↓

E(3)-Equivariant Message Passing

↓

Atomic Feature

↓

Energy

↓

Force
```

이다.

---

# What is E(3)?

E(3)는

3차원 공간의 대칭성을 의미한다.

포함되는 연산은

- Translation
- Rotation
- Reflection

이다.

좋은 Potential은

이 연산에 대해

물리적으로 올바르게 동작해야 한다.

---

# Invariant vs Equivariant

기존 방법

```
Rotate Structure

↓

Feature

=

Same
```

(Invariant)

NequIP

```
Rotate Structure

↓

Feature

↓

Rotate Together
```

(Equivariant)

즉

회전 정보를 버리지 않는다.

이것이 가장 큰 차별점이다. :contentReference[oaicite:1]{index=1}

---

# Graph Neural Network

원자는 Node

결합은 Edge

이다.

```
Atom

↓

Node

↓

Neighbor Message

↓

Update

↓

New Feature
```

Graph Message Passing을 여러 번 수행하면서

원자 주변 환경을 점점 정교하게 표현한다.

---

# Equivariant Message Passing

일반 GNN은

Scalar Feature만 전달한다.

NequIP은

```
Scalar

+

Vector

+

Higher-order Tensor
```

를 함께 전달한다.

즉

방향성을 가진 Feature도

학습된다.

---

# Spherical Harmonics

방향 정보는

Spherical Harmonics를 이용하여 표현한다.

```
Distance

↓

Radial Function

Direction

↓

Spherical Harmonics

↓

Tensor Product

↓

Interaction
```

따라서

방향 정보가 그대로 유지된다. :contentReference[oaicite:2]{index=2}

---

# Energy Prediction

각 원자의 Energy는

```
Ei

=

NN(Graph Feature)
```

전체 Energy는

```
E

=

Σ Ei
```

Force는

```
F

=

−∂E/∂R
```

를 이용하여 계산한다.

---

# Why Data Efficient?

논문의 가장 큰 성과는

Data Efficiency이다.

기존 Neural Network들은

수만 개의 DFT 데이터가 필요했다.

NequIP은

수백~수천 개의 DFT 데이터만으로도

기존 모델과 동일하거나 더 높은 정확도를 달성하였다.

일부 데이터셋에서는

약 **100~1,000개의 DFT 구조만으로도 높은 정확도**를 보였으며, 기존 모델 대비 최대 **세 자릿수 수준의 데이터 절감**을 달성하였다. :contentReference[oaicite:3]{index=3}

---

# Performance

평가 데이터

- MD17
- Water
- Amorphous Materials
- Lithium Conductor
- Surface Reaction

모든 데이터셋에서

당시 SOTA 성능을 달성하였다.

특히

Force Prediction에서

매우 높은 정확도를 보였다. :contentReference[oaicite:4]{index=4}

---

# Why This Paper Matters

NequIP 이후

MLIP 연구는

Descriptor Engineering에서

Geometry Learning으로 방향이 바뀌었다.

후속 연구

- Allegro
- MACE
- SevenNet
- Orb
- MatterSim

거의 모두

E(3)-Equivariant 구조를 사용한다.

---

# Semiconductor Relevance

반도체에서는

다음과 같은 계산이 중요하다.

- Crystal Defect
- Grain Boundary
- Interface
- Dopant Diffusion
- Vacancy
- Stress
- Phase Transition

이들은

원자의 방향성이 매우 중요하다.

따라서

NequIP은

Si

SiGe

GaN

Oxide

HBM 소재

등의 Material Simulation에서

높은 정확도를 기대할 수 있다.

---

# Relation with SK hynix TCAD

SK hynix JD의

```
DFT

↓

NequIP

↓

Large Scale MD

↓

Extract Material Properties

↓

TCAD Parameter

↓

Device Simulation
```

이라는

Physics AI Pipeline과

직접 연결된다.

특히

NequIP은

Material Simulation을

AI로 가속하는 대표적인 사례이다.

---

# Limitations

## 1. 계산량이 크다.

Equivariant Convolution은

일반 GNN보다

계산량이 훨씬 많다.

---

## 2. 추론 속도

정확도는 높지만

추론 속도는

Allegro보다 느리다.

---

## 3. GPU Memory

Higher-order Tensor를 사용하므로

메모리 사용량이 크다.

---

## 4. 대규모 MD에는 부담

수백만 원자 규모에서는

후속 모델(Allegro, MACE)이

더 적합한 경우가 많다.

---

# DeepPot-SE vs NequIP

| 항목 | DeepPot-SE | NequIP |
|------|------------|---------|
| 입력 | Learned Descriptor | Graph |
| Feature | Scalar | Scalar + Vector + Tensor |
| 대칭성 | Invariant | E(3)-Equivariant |
| 방향 정보 | 일부 손실 | 유지 |
| 데이터 효율 | 높음 | 매우 높음 |
| 정확도 | 매우 우수 | 당시 SOTA |

---

# My Insight

NequIP을 공부하면서 가장 크게 느낀 점은

**"좋은 Physics AI는 물리 정보를 제거하는 것이 아니라 보존하는 방향으로 발전한다."**

DeepPot-SE가 Descriptor를 잘 설계하는 시대였다면,

NequIP은 **3차원 기하학 자체를 학습하는 시대**를 열었다.

이후 등장한 Allegro와 MACE도 모두 이 철학을 계승하고 있으며,

현재 MLIP 연구의 중심은 **Equivariant GNN**이라고 볼 수 있다.

TCAD 관점에서도

원자 수준 Material Simulation을 AI로 대체하는 핵심 기술 중 하나라고 생각된다.

---

# Keywords

- NequIP
- E(3)-Equivariant GNN
- Graph Neural Network
- Message Passing
- Spherical Harmonics
- Machine Learning Interatomic Potential
- MLIP
- Molecular Dynamics
- Materials Informatics
- Physics AI