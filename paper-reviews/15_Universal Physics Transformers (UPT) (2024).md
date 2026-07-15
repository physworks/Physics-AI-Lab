# Universal Physics Transformers (UPT)

> Benedikt Alkin, Andreas Fürst, Simon Schmid, Lukas Gruber, Markus Holzleitner, Johannes Brandstetter (2024)
>
> **Universal Physics Transformers: A Framework for Efficiently Scaling Neural Operators**
>
> NeurIPS 2024

---

## Why I Read This Paper

PINN은 하나의 PDE를 학습하고,

DeepONet과 Fourier Neural Operator(FNO)는 PDE Operator를 학습한다.

MeshGraphNet은 Mesh 기반 Simulation을 학습한다.

하지만 각각은

- Grid 전용
- Mesh 전용
- Particle 전용

등 특정 Simulation 방식에 특화되어 있다.

실제 반도체 Virtual R&D에서는

- Material Simulation
- Molecular Dynamics
- TCAD
- CFD
- FEM
- Particle Simulation

등 다양한 형태의 시뮬레이션이 함께 사용된다.

이 논문는 이러한 문제를 해결하기 위해

**모든 물리 시뮬레이션을 하나의 Transformer Architecture로 학습하는 Universal Physics Transformer(UPT)를 제안한다.**

즉,

Scientific Machine Learning의 GPT와 같은 Foundation Model을 만들려는 시도이다. :contentReference[oaicite:1]{index=1}

---

## Problem

현재 Scientific Machine Learning 모델들은

```
PINN

↓

특정 PDE
```

```
FNO

↓

Grid 기반
```

```
MeshGraphNet

↓

Mesh 기반
```

처럼

Problem-specific Architecture이다.

따라서

새로운 Simulation마다

새로운 Network를 설계해야 한다.

이는

Large-scale Physics Foundation Model을 만드는 데 가장 큰 장애물이다.

---

## Key Idea

논문의 핵심은

**Physics Domain과 Discretization에 관계없는 Universal Transformer를 만드는 것이다.**

기존

```
Grid

↓

Grid Network
```

```
Mesh

↓

Graph Network
```

```
Particle

↓

Particle Network
```

UPT는

```
Physics Data

↓

Universal Encoder

↓

Latent Physics Space

↓

Transformer

↓

Universal Decoder

↓

Prediction
```

으로

모든 Simulation을 하나의 Latent Space에서 처리한다.

즉,

Simulation 종류보다

Physics 자체를 학습한다.

---

## Method

### Step 1. Universal Encoding

입력 데이터가

- Structured Grid
- Unstructured Mesh
- Particle

어떤 형태이든

모두

공통 Latent Space로 변환한다.

```
Grid

↓

Encoder

↓

Latent
```

```
Mesh

↓

Encoder

↓

Latent
```

```
Particle

↓

Encoder

↓

Latent
```

---

### Step 2. Latent Transformer

Transformer는

Physics State들의 관계를

Self-Attention으로 학습한다.

```
Latent Physics

↓

Transformer

↓

Updated Physics
```

Grid인지

Mesh인지

Particle인지

구분하지 않는다.

---

### Step 3. Inverse Decoder

Latent Representation을

원래 Simulation 형태로 복원한다.

```
Latent

↓

Decoder

↓

Simulation Result
```

필요한 위치에서만 값을 질의(Query)할 수 있어

매우 효율적이다.

---

### Step 4. Time Rollout

Transformer를 반복 적용하여

```
t

↓

t+1

↓

t+2

↓

...

↓

t+n
```

Physics Evolution을 수행한다.

---

## Equation

UPT는

입력 Physics State

\[
x
\]

를

Latent Space

\[
z
\]

로 변환한다.

\[
z = E(x)
\]

Transformer는

\[
z' = T(z)
\]

를 계산한다.

최종 Prediction은

\[
\hat y=D(z')
\]

이다.

즉,

```
Encoder

↓

Transformer

↓

Decoder
```

구조이다.

Loss는

\[
L
=
MSE(y,\hat y)
\]

를 최소화한다.

필요에 따라

Physics Constraint도 함께 적용할 수 있다. :contentReference[oaicite:2]{index=2}

---

## Advantages

- Grid, Mesh, Particle를 모두 처리 가능
- 하나의 Architecture로 다양한 Physics 문제 해결
- Neural Operator의 확장성 향상
- Transformer 기반으로 대규모 학습 가능
- Foundation Model 구축 가능
- 다양한 Simulation 간 Transfer Learning 가능
- 차세대 Virtual R&D 플랫폼에 적합

---

## Limitations

- 매우 큰 학습 데이터셋이 필요하다.
- Transformer의 계산 비용이 높다.
- Latent Space 설계가 성능에 큰 영향을 준다.
- 아직 TCAD에 직접 적용된 사례는 많지 않다.
- 산업 현장에서는 추가적인 검증이 필요하다.

---

## Key Takeaways

- UPT는 Physics Foundation Model을 목표로 한다.
- Simulation 종류가 아니라 Physics 자체를 학습한다.
- Grid, Mesh, Particle를 하나의 모델로 통합한다.
- Scientific Machine Learning의 GPT와 같은 역할을 지향한다.
- 향후 AI 기반 Virtual R&D의 핵심 기술이 될 가능성이 높다.

---

## Personal Notes

### Relation to My Current Project

현재 GitHub 프로젝트는

```
GCA Equation

↓

PINN

↓

Device Prediction
```

이다.

향후에는

```
TCAD

↓

Neural Operator

↓

Universal Physics Transformer

↓

Physics Foundation Model
```

까지 확장하고 싶다.

---

### Relation to My Career

UPT는

```
Any Physics

↓

Universal AI

↓

Simulation
```

으로 확장한 개념이다.

기존 경험을 더 넓은 Scientific Machine Learning 영역으로 연결할 수 있다.

---

### Relation to SK hynix TCAD

SK하이닉스 JD의

- Material Simulation
- Physics AI
- Scientific Machine Learning
- Virtual R&D

를 하나의 Architecture로 연결하는 방향성과 매우 유사하다.

장기적으로는

```
MLIP

↓

Molecular Dynamics

↓

TCAD

↓

Compact Model

↓

Circuit

↓

System
```

전체를 하나의 Foundation Model로 연결하는 기반이 될 가능성이 있다.

---

### Comparison with Previous Papers

| Paper | 입력 형태 | 학습 대상 | 특징 |
|---------|------------|----------------------|----------------------------|
| PINN | Point | PDE Solution | Physics Loss |
| DeepONet | Function | Operator | Branch + Trunk |
| FNO | Grid | Operator | Fourier Domain |
| MeshGraphNet | Mesh | Simulation | Graph Network |
| MLIP | Atomic | Interatomic Potential | DFT Surrogate |
| **UPT** | Grid + Mesh + Particle | Universal Physics | Physics Foundation Model |

UPT는 지금까지 리뷰한 논문들을 하나의 Framework로 통합하려는 첫 번째 시도라고 볼 수 있다.

---

### Relation to TCAD

향후 AI 기반 반도체 연구는

```
Atomic Structure

↓

MLIP

↓

Material Property

↓

TCAD

↓

UPT

↓

Virtual R&D Platform
```

으로 발전할 가능성이 높다.

TCAD도 더 이상 독립적인 Solver가 아니라

Physics Foundation Model의 일부가 될 것으로 예상된다.

---

### Future Learning Plan

이 논문를 기반으로 다음 프로젝트를 진행할 계획이다.

- [ ] Transformer 기반 Neural Operator 구현
- [ ] Latent Physics Representation 학습
- [ ] UPT 구조 분석
- [ ] TCAD Dataset 적용
- [ ] Multi-Physics Dataset 구축
- [ ] Physics Foundation Model 프로젝트 진행

---

### Paper Link

https://arxiv.org/abs/2402.12365