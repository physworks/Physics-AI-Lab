# Paper Name

> Han Wang et al. (2018)
>
> **DeePMD-kit: A Deep Learning Package for Many-Body Potential Energy Representation and Molecular Dynamics**
>
> Computer Physics Communications, 228, 178-184 (2018)

---

# Why I Read This Paper

SK hynix TCAD JD에서 요구하는 **Material Simulation (DFT/MD/KMC)** 과 **Physics AI**를 공부하다 보면 반드시 등장하는 것이 **Machine Learning Interatomic Potential (MLIP)** 이다.

DeePMD-kit은 현재 가장 널리 사용되는 MLIP 프레임워크 중 하나이며,

- Deep Potential(DP)
- MACE
- NequIP
- Allegro

등 최신 MLIP 연구의 출발점이라고 볼 수 있다.

반도체에서는

- 원자 수준 Diffusion
- Defect
- Grain Boundary
- Stress
- Interface

등을 DFT보다 훨씬 빠르게 계산하기 위해 사용된다.

---

# Background

기존 Molecular Dynamics(MD)는 크게 두 가지가 존재한다.

## Classical MD

- Lennard-Jones
- EAM
- Tersoff
- Stillinger-Weber

장점

- 매우 빠르다.

단점

- 정확도가 낮다.

---

## Ab-initio MD (AIMD)

DFT를 이용하여

매 timestep마다

Electronic Structure를 계산한다.

장점

- 매우 정확하다.

단점

- 너무 느리다.

대부분

100~1000 atoms 정도밖에 계산하지 못한다.

---

결국

Accuracy와 Speed는 항상 Trade-off였다.

```
Classical MD
Fast
Poor Accuracy

↓

DeepMD

↓

DFT
Slow
High Accuracy
```

DeepMD는

이 두 세계를 연결하려는 연구이다. :contentReference[oaicite:0]{index=0}

---

# Core Idea

핵심 아이디어는 매우 단순하다.

> DFT가 계산한 Energy Surface를
>
> Neural Network가 학습하자.

즉,

```
Atomic Coordinates

↓

Neural Network

↓

Potential Energy

↓

Force
```

Force는

```
F = -∂E/∂R
```

를 이용하여 자동미분으로 계산한다.

이렇게 하면

DFT 수준 정확도를 유지하면서

MD 속도는 Classical MD 수준으로 만들 수 있다. :contentReference[oaicite:1]{index=1}

---

# Deep Potential Model

입력은

원자의 위치이다.

하지만

단순 Cartesian Coordinate를 넣으면

회전하거나 이동할 때

결과가 달라진다.

따라서

DeePMD는

Local Coordinate를 만든다.

```
Atom i

↓

Neighbor Search

↓

Local Environment

↓

Descriptor

↓

Neural Network

↓

Atomic Energy
```

전체 에너지는

```
E_total

=

Σ Ei
```

이다.

---

# Important Property

Descriptor는

다음 성질을 만족해야 한다.

## Translation Invariant

전체를 이동해도

에너지는 동일

---

## Rotation Invariant

회전해도 동일

---

## Permutation Invariant

같은 종류 원자를 교환해도

동일

---

이 성질이

Deep Potential의 핵심이다.

---

# Workflow

전체 Workflow는

```
DFT

↓

Energy
Force
Virial

↓

Training Dataset

↓

DeePMD-kit

↓

Deep Potential

↓

LAMMPS

↓

Large Scale MD
```

이다.

즉

DFT는

Training 데이터 생성용으로만 사용된다.

실제 MD는

Neural Network가 수행한다. :contentReference[oaicite:2]{index=2}

---

# DeePMD-kit

논문의 핵심은

Deep Potential 알고리즘 자체보다

이를 사용할 수 있는

Software Package를 제공했다는 점이다.

지원 기능

- TensorFlow 기반 학습
- LAMMPS 연동
- i-PI 연동
- Energy 학습
- Force 학습
- Virial 학습
- GPU 지원

현재는

PyTorch, JAX 등도 지원하는 v3까지 발전하였다. :contentReference[oaicite:3]{index=3}

---

# Why Force Learning?

Energy만 학습하면

Gradient가 틀릴 수 있다.

반면

Force도 함께 학습하면

Potential Surface 전체를

훨씬 정확하게 복원할 수 있다.

Loss는 보통

```
Loss

=

wE LE

+

wF LF

+

wV LV
```

이다.

즉

Energy

Force

Virial을

동시에 학습한다.

---

# Performance

논문에서는

Water DFT Dataset을 이용하여

Deep Potential을 학습하였다.

결과

- RDF 재현
- Structure 재현
- Thermodynamic Property 재현

모두

DFT와 거의 동일하였다.

하지만

계산 속도는

DFT보다 수천~수만 배 빠르다. :contentReference[oaicite:4]{index=4}

---

# Why This Paper Matters

이 논문 이후

MLIP 시대가 시작되었다.

대표적인 후속 연구

- DeepPot-SE
- DPLR
- DPRc
- MACE
- NequIP
- Allegro
- CHGNet
- MatterSim

거의 모두가

Deep Potential의 영향을 받았다.

---

# Semiconductor Relevance

반도체에서는

다음과 같은 계산을 수행한다.

- Si Diffusion
- Vacancy
- Interstitial
- Grain Boundary
- Interface
- Oxidation
- Dopant Migration

기존에는

DFT 때문에

계산량이 너무 컸다.

하지만

DeepMD를 이용하면

수백만 원자 규모까지

시뮬레이션이 가능하다.

따라서

TCAD의 Material Parameter 생성에도

활용 가능성이 높다.

---

# Relation with SK hynix TCAD

JD의

Material Simulation과 연결하면

```
DFT

↓

DeepMD

↓

Large Scale MD

↓

Material Property

↓

TCAD Parameter

↓

Device Simulation
```

이라는 흐름이 만들어진다.

즉

DeepMD는

Physics AI의 첫 번째 단계라고 볼 수 있다.

---

# Limitations

## 1. DFT Dataset 의존

좋은 Dataset이 없으면

성능도 낮다.

---

## 2. Extrapolation 문제

학습하지 않은 구조에서는

오차가 커질 수 있다.

---

## 3. Long-range Interaction

초기 버전은

Electrostatic Interaction 표현이 부족했다.

후속 연구인

DPLR에서 개선하였다.

---

## 4. Training Cost

대규모 DFT 데이터 생성 비용이

여전히 크다.

---

# My Insight

이 논문을 보면서

DeepMD는 단순히 MD를 빠르게 하는 알고리즘이 아니라,

**"Physics Simulator를 AI Surrogate로 치환한 최초의 성공 사례"**라는 점이 가장 인상 깊었다.

특히

```
Physics Simulation

↓

Generate Dataset

↓

Train Neural Network

↓

Replace Physics Solver
```

라는 패러다임은

현재

- Neural Operator
- PINN
- TCAD Surrogate
- AI SPICE
- Physics Foundation Model

모두의 출발점이라고 생각된다.

앞으로 SK hynix TCAD에서 추진하는 Physics AI 역시

결국 이러한 방향으로 발전할 가능성이 높다고 판단된다.

---

# Keywords

- Machine Learning Potential (MLIP)
- Deep Potential
- Molecular Dynamics
- Ab-initio Molecular Dynamics
- DFT
- LAMMPS
- Force Learning
- Interatomic Potential
- Materials Informatics
- Physics AI