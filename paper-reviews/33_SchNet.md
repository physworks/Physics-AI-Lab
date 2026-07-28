# Paper Name

> Kristof T. Schütt et al. (2017)
>
> **SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions**
>
> NeurIPS 2017

---

# Why I Read This Paper

최근 MLIP(Machine Learning Interatomic Potential)의 발전은

```
SchNet

↓

DeepMD

↓

NequIP

↓

Allegro

↓

MACE
```

와 같이 이어진다.

SchNet은

오늘날 대부분의 Atomistic AI의 출발점이라고 할 수 있는 논문이다.

특히

- Continuous-filter Convolution
- End-to-end Learning
- Learned Atomic Embedding

이라는 개념을 처음 성공적으로 도입하여

원자 구조를 CNN으로 직접 학습할 수 있음을 보여주었다. :contentReference[oaicite:0]{index=0}

---

# Background

기존 분자 머신러닝은

사람이 Descriptor를 설계하였다.

예를 들어

- Coulomb Matrix
- ACSF
- SOAP

등을 입력으로 사용하였다.

즉

```
Atomic Structure

↓

Hand-crafted Descriptor

↓

Machine Learning
```

이었다.

하지만

Descriptor를 잘못 설계하면

성능이 크게 떨어진다.

SchNet은

> Descriptor를 사람이 만들지 말고 AI가 직접 배우자.

라는 아이디어를 제안하였다. :contentReference[oaicite:1]{index=1}

---

# Core Idea

SchNet의 핵심은

**Continuous-filter Convolution(CFConv)** 이다.

기존 CNN은

```
Image

↓

3×3 Filter

↓

Feature Map
```

처럼

격자(Grid) 위에서만 동작한다.

하지만

원자는

격자 위에 존재하지 않는다.

```
Random Atomic Position

↓

?

↓

CNN 불가능
```

SchNet은

거리 정보를 이용하여

연속적인 Filter를 생성하였다.

```
Atomic Position

↓

Distance

↓

Continuous Filter

↓

Convolution
```

이를 통해

비정형 원자 구조에서도

Convolution을 수행할 수 있다. :contentReference[oaicite:2]{index=2}

---

# Continuous Filter Convolution

기존 CNN

```
Fixed Kernel

↓

Feature
```

SchNet

```
Distance

↓

MLP

↓

Continuous Filter

↓

Feature
```

즉

Filter 자체를

거리의 함수로 만든다.

따라서

원자의 위치가 조금 변해도

부드럽게 반응한다.

---

# Atomic Embedding

각 원자는

원자 번호(Z)를 입력받는다.

```
Atomic Number

↓

Embedding

↓

Atomic Feature
```

예를 들어

```
H

↓

Vector

C

↓

Vector

Si

↓

Vector
```

처럼

각 원소는

학습 가능한 벡터로 변환된다.

모델은

주기율표의 화학적 유사성까지

스스로 학습한다. :contentReference[oaicite:3]{index=3}

---

# Interaction Block

SchNet의 핵심 연산은

Interaction Block이다.

```
Atom Feature

↓

Continuous Filter

↓

Neighbor Interaction

↓

Updated Feature
```

이 과정을

여러 번 반복한다.

즉

Graph Neural Network의

Message Passing과

매우 유사한 구조이다.

실제로 SchNet은 현대 Message Passing Neural Network(MPNN)의 초기 형태로 평가받는다. :contentReference[oaicite:4]{index=4}

---

# Energy Prediction

최종적으로

각 원자의 Energy를 계산한다.

```
Ei

=

MLP(Feature_i)
```

전체 Energy는

```
E

=

Σ Ei
```

이다.

Force는

자동미분을 이용하여

```
F = -∂E/∂R
```

로 계산한다.

따라서

Energy와 Force를

동시에 학습할 수 있다. :contentReference[oaicite:5]{index=5}

---

# Why This Paper Matters

SchNet 이전에는

대부분

Descriptor Engineering이 중심이었다.

SchNet 이후에는

Representation Learning이 중심이 되었다.

즉

```
Hand-crafted Descriptor

↓

Learned Representation
```

이라는

큰 패러다임 전환이 일어났다.

---

# Performance

논문에서는

다음 데이터셋에서 평가하였다.

- QM9
- MD17

결과

- 당시 State-of-the-art
- 높은 Force Prediction 정확도
- 우수한 Molecular Property Prediction

을 달성하였다. :contentReference[oaicite:6]{index=6}

---

# Semiconductor Relevance

반도체 Material Simulation에서는

다음 문제가 중요하다.

- Silicon
- Germanium
- Oxide
- Interface
- Vacancy
- Dopant

SchNet은

원자 구조에서

직접 Feature를 학습하기 때문에

향후

- DFT Surrogate
- Force Field
- Material Property Prediction

등에 활용될 수 있다.

---

# Relation with SK hynix TCAD

Material Simulation Pipeline은

```
DFT

↓

SchNet

↓

ML Force Field

↓

Large-scale MD

↓

Material Property

↓

TCAD Parameter

↓

Device Simulation
```

으로 연결된다.

즉

SchNet은

Physics AI 기반 Material Simulation의

초기 대표 모델이다.

---

# Limitations

## 1. Direction 정보를 충분히 활용하지 못함

SchNet은

거리(distance)를 중심으로 학습하기 때문에

방향(orientation) 정보 활용에는 한계가 있다.

이 문제는

NequIP의 E(3)-Equivariant GNN에서 크게 개선되었다.

---

## 2. Data Efficiency

SchNet은

최신 MLIP보다

더 많은 학습 데이터가 필요한 경우가 많다.

---

## 3. Many-body 표현력

Higher-order Interaction 표현력이

MACE보다 낮다.

---

## 4. Scalability

초대규모 MD에서는

Allegro가

더 높은 계산 효율을 제공한다.

---

# SchNet vs DeepPot vs NequIP

| 항목 | SchNet | DeepPot-SE | NequIP |
|------|---------|------------|---------|
| Descriptor | Learned | Learned | Geometry 기반 |
| Continuous Filter | O | X | X |
| Message Passing | O | X | O |
| Equivariant | X | X | O |
| 방향 정보 | 제한적 | 제한적 | 완전 보존 |
| 발표 | 2017 | 2018 | 2021 |

---

# My Insight

SchNet을 공부하면서 가장 인상 깊었던 점은

**"원자도 이미지처럼 Convolution으로 학습할 수 있다."**는 발상이었다.

물론 현재 기준에서는

NequIP, Allegro, MACE보다 성능이 낮지만,

SchNet이 없었다면

Continuous Filter,

Message Passing,

Graph Neural Network 기반 MLIP의 발전도 훨씬 늦어졌을 것이다.

Physics AI의 역사에서

SchNet은

**Descriptor Engineering 시대를 끝내고 Representation Learning 시대를 연 논문**이라고 생각된다.

---

# Keywords

- SchNet
- Continuous-filter Convolution
- CFConv
- Machine Learning Interatomic Potential
- MLIP
- Molecular Dynamics
- Message Passing
- Graph Neural Network
- Materials Informatics
- Physics AI