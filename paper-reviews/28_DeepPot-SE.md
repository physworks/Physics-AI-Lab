# Paper Name

> Linfeng Zhang et al. (2018)
>
> **End-to-end Symmetry Preserving Inter-atomic Potential Energy Model for Finite and Extended Systems (DeepPot-SE)**
>
> NeurIPS 2018

---

# Why I Read This Paper

DeePMD-kit을 공부한 뒤 가장 궁금했던 점은 다음이었다.

> "원자 주변 환경(Descriptor)을 어떻게 표현해야 다양한 물질에서도 정확한 Potential을 학습할 수 있을까?"

DeepPot-SE(Smooth Edition)는 바로 이 문제를 해결한 논문이다.

기존 Deep Potential을 개선하여

- 더 부드러운(Smooth) Descriptor
- 더 안정적인 Force 계산
- 다양한 물질(금속, 반도체, 절연체, 유기분자)에 대한 높은 일반화 성능

을 달성했으며, 오늘날 DeePMD-kit의 기본 모델로 사용되고 있다.

---

# Background

Interatomic Potential은 다음 조건을 만족해야 한다.

- Translation Invariance
- Rotation Invariance
- Permutation Invariance
- Continuous Differentiability
- Linear Scaling

초기 Deep Potential은 이러한 조건을 대부분 만족했지만, Neighbor가 Cutoff 경계를 넘나들 때 Descriptor가 불연속적으로 변하는 문제가 있었다.

이는 Molecular Dynamics에서 Force의 불안정성을 유발할 수 있었다.

DeepPot-SE는 이 문제를 **Smooth Descriptor**로 해결하였다. :contentReference[oaicite:0]{index=0}

---

# Core Idea

핵심 아이디어는

> **Neighbor 정보를 부드럽게(Smooth) Encoding하여 항상 연속적인 Potential Surface를 만들자.**

전체 구조는

```
Atomic Coordinates

↓

Neighbor List

↓

Smooth Descriptor

↓

Embedding Network

↓

Fitting Network

↓

Atomic Energy

↓

Total Energy

↓

Force
```

이다.

---

# Smooth Edition

기존 모델에서는

```
Neighbor enters cutoff

↓

Descriptor 급격한 변화

↓

Force 불연속
```

문제가 발생하였다.

DeepPot-SE에서는

Cutoff 부근에 Smooth Function을 적용한다.

```
Neighbor enters cutoff

↓

Smooth Weight

↓

Descriptor 연속

↓

Force 연속
```

이것이

SE(Smooth Edition)의 가장 중요한 아이디어이다. :contentReference[oaicite:1]{index=1}

---

# Descriptor

각 원자 i에 대해

주변 원자를 검색한다.

```
Atom i

↓

Neighbor Search

↓

Relative Position

↓

Embedding Network

↓

Descriptor Matrix
```

여기서

각 Neighbor 정보를

Neural Network가 자동으로 Embedding한다.

즉

사람이 Feature를 설계하지 않는다.

---

# Embedding Network

기존 ACSF(Atom-centered Symmetry Function)는

사람이 Descriptor를 설계해야 했다.

DeepPot-SE는

```
Relative Coordinate

↓

Embedding Network

↓

Learned Descriptor
```

방식을 사용한다.

즉

Descriptor 자체도

학습된다.

이것이 기존 MLIP 대비 가장 큰 차별점이다.

---

# Energy Model

각 원자의 Energy는

```
Ei

=

NN(Descriptor_i)
```

전체 Energy는

```
E

=

Σ Ei
```

이다.

Force는

```
F

=

−∂E/∂R
```

를 이용하여

Automatic Differentiation으로 계산한다.

---

# Why Smooth Matters

MD에서는

Force가 조금만 흔들려도

Simulation이 불안정해진다.

따라서

Potential은

반드시

Continuous해야 한다.

DeepPot-SE는

Energy뿐 아니라

Force까지

매끄럽게 만든 것이 핵심이다.

---

# Performance

논문에서는

다양한 시스템에서 평가하였다.

- Organic Molecule
- Copper
- Silicon
- Water
- Alloy
- High Entropy Alloy

모든 데이터셋에서

기존 Descriptor 기반 방법과 비교하여 높은 정확도를 보였으며,

특히 Bulk Material과 반도체 시스템에서 우수한 성능을 보였다. :contentReference[oaicite:2]{index=2}

---

# Why This Paper Matters

DeepPot-SE는 이후 Deep Potential 계열 연구의 표준이 되었다.

후속 연구

- DeePMD-kit v2
- DPLR
- DPRc
- DPGEN
- Deep Potential Long Range
- Deep Potential Compression

모두 DeepPot-SE를 기반으로 발전하였다. :contentReference[oaicite:3]{index=3}

---

# Semiconductor Relevance

반도체 Material Simulation에서는

다음과 같은 계산이 필요하다.

- Silicon
- SiGe
- Oxide
- Interface
- Vacancy
- Diffusion
- Stress
- Defect

DeepPot-SE는

DFT 수준 정확도를 유지하면서

대규모 MD를 수행할 수 있으므로

다음과 같은 흐름이 가능하다.

```
DFT

↓

DeepPot-SE

↓

Large Scale MD

↓

Material Property

↓

TCAD Parameter

↓

Device Simulation
```

즉

TCAD Material Parameter 생성의 핵심 기술이 될 수 있다.

---

# Relation with SK hynix TCAD

JD에서 언급된

- Material Simulation
- Physics AI
- AI-assisted TCAD

와 연결하면

```
DFT

↓

DeepPot-SE

↓

Atomic Simulation

↓

Extract Material Parameters

↓

TCAD

↓

Device Simulation
```

이라는 파이프라인이 완성된다.

향후 TCAD Surrogate, PINN, Neural Operator와도 자연스럽게 연결된다.

---

# Limitations

## 1. DFT Dataset 품질에 의존

좋은 데이터가 없으면 성능도 제한된다.

---

## 2. Long-range Interaction

초기 DeepPot-SE는 장거리 Coulomb Interaction 표현이 제한적이다.

후속 연구인 DPLR에서 이를 개선하였다.

---

## 3. Extrapolation

훈련 데이터에서 크게 벗어난 구조에서는 정확도가 떨어질 수 있다.

---

## 4. 대규모 학습 비용

Embedding Network와 Fitting Network를 함께 학습하므로 초기 학습 비용이 크다.

---

# DeepPot vs DeepPot-SE

| 항목 | Deep Potential | DeepPot-SE |
|------|---------------|------------|
| Descriptor | 초기 버전 | Smooth Descriptor |
| Force 연속성 | 제한적 | 우수 |
| Cutoff 처리 | 불연속 가능 | Smooth |
| 일반화 | 보통 | 우수 |
| 현재 사용 | 거의 사용 안 함 | 표준 모델 |

---

# My Insight

DeepPot-SE의 가장 큰 기여는 단순히 정확도를 높인 것이 아니라,

**"물리 법칙(대칭성)과 딥러닝을 함께 설계했다"**는 점이다.

특히 Descriptor를 사람이 설계하는 대신 Neural Network가 학습하면서도,

회전·병진·원자 교환 대칭성과 연속성을 유지하도록 설계한 것은 이후 **NequIP, Allegro, MACE**와 같은 최신 E(3)-Equivariant 모델로 이어지는 중요한 전환점이 되었다.

Physics AI의 관점에서 보면 DeepPot-SE는 "Symmetry-aware MLIP" 시대를 연 대표적인 논문이라고 생각된다.

---

# Keywords

- DeepPot-SE
- Smooth Descriptor
- Machine Learning Interatomic Potential
- MLIP
- Molecular Dynamics
- DFT
- Symmetry Preserving
- Force Learning
- Materials Informatics
- Physics AI