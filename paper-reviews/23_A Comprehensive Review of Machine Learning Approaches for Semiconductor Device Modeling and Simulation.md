# Paper Name

> Mamun, Pala, Shawkat (2025)
>
> A Comprehensive Review of Machine Learning Approaches for Semiconductor Device Modeling and Simulation
>
> IEEE Access

---

## Why I Read This Paper

최근 반도체 산업에서는 Machine Learning을 이용하여 TCAD Simulation, Compact Model, Device Parameter Extraction, Process Optimization을 가속화하는 연구가 매우 활발하게 진행되고 있다.

이 논문은 이러한 연구들을 하나의 프레임워크로 정리한 **최신 Survey 논문**으로,

- TCAD + AI
- Physics AI
- PINN
- Compact Model
- Device Modeling
- Surrogate Model

등 현재 Semiconductor AI 분야를 전체적으로 이해하기 위해 읽었다. :contentReference[oaicite:0]{index=0}

---

## Problem

반도체 소자는

- 공정 변수 증가
- 구조 복잡성 증가
- 새로운 소재 도입
- 3D Device 확대

등으로 인해

TCAD Simulation 시간이 매우 길어지고 있다.

특히

- Device Optimization
- Parameter Sweep
- Yield Prediction
- Compact Modeling

에서는 수천~수만 번의 Simulation이 필요하기 때문에

계산 비용이 급격히 증가한다.

또한 기존 Physics 기반 모델은

- 정확하지만 느리고
- 새로운 Device마다 재모델링이 필요하며
- 데이터 활용이 제한적이라는 문제가 있다.

Machine Learning은 이러한 문제를 해결하여

빠른 Device Prediction과 Virtual R&D를 가능하게 한다. :contentReference[oaicite:1]{index=1}

---

## Key Idea

논문은 Machine Learning 기반 Semiconductor Modeling을 크게 6개 분야로 분류한다.

```
Semiconductor AI

├── Device Modeling
├── TCAD Acceleration
├── Compact Modeling
├── Parameter Extraction
├── Process Optimization
└── Physics-informed AI
```

Machine Learning은

복잡한 Device Physics를

완전히 대체하는 것이 아니라

Physics Solver를 보완하거나

Surrogate Model을 구축하는 방향으로 발전하고 있다. :contentReference[oaicite:2]{index=2}

---

## Method

### 1. Device Modeling

가장 많이 활용되는 분야이다.

입력

- Gate Length
- Oxide Thickness
- Doping
- Temperature
- Geometry

↓

ML

↓

출력

- Id-Vg
- Id-Vd
- Threshold Voltage
- Leakage
- Mobility

복잡한 Device 특성을 매우 빠르게 예측한다.

---

### 2. Compact Modeling

SPICE용 Compact Model을

Neural Network로 대체한다.

```
TCAD

↓

Training Data

↓

Neural Network

↓

Compact Model
```

기존 BSIM 기반 모델보다

빠른 모델 생성이 가능하다.

---

### 3. TCAD Surrogate Model

가장 중요한 분야 중 하나이다.

```
TCAD

↓

Dataset

↓

Machine Learning

↓

Fast Prediction
```

수 시간~수 일 걸리는 Simulation을

초 단위 Prediction으로 대체한다.

최근

- PINN
- Neural Operator
- Graph Network

등이 활발히 연구되고 있다.

---

### 4. Parameter Extraction

Device Parameter

예)

- Mobility
- Dit
- Oxide Charge
- Contact Resistance

등을

Machine Learning으로 역추정한다.

이는 기존 Optimization보다

매우 빠르다.

---

### 5. Process Optimization

공정 변수

- Implant
- Annealing
- Oxidation
- Deposition

등을 입력으로 사용하여

Device 특성을 예측한다.

```
Process

↓

ML

↓

Device Performance
```

공정 최적화 시간을 크게 줄일 수 있다.

---

### 6. Physics-informed AI

최근 가장 활발한 연구 분야이다.

대표적으로

- PINN
- Physics-informed GNN
- Physics Neural Operator

등이 소개된다.

Machine Learning이

Physics Constraint를 만족하도록 학습한다. :contentReference[oaicite:3]{index=3}

---

## Equation

대표적인 Device Mapping은

\[
y=f(x)
\]

여기서

Input

\[
x=
(Device,\ Process,\ Material)
\]

Output

\[
y=
(I_d,\ V_{th},\ Leakage,\ Mobility)
\]

PINN에서는

\[
L
=
L_{data}
+
L_{physics}
\]

Neural Operator에서는

\[
u=G(a)
\]

Graph Network에서는

\[
h_i^{k+1}
=
f
(h_i,\mathcal N(i))
\]

등 다양한 모델이 사용된다. :contentReference[oaicite:4]{index=4}

---

## Advantages

- TCAD Simulation 시간 단축
- Device Optimization 가속
- Parameter Extraction 자동화
- Compact Model 구축 시간 단축
- 새로운 Device 구조에도 적용 가능
- 복잡한 비선형 특성 학습 가능
- Virtual R&D 구현 가능
- 산업 적용 사례가 빠르게 증가하고 있다. :contentReference[oaicite:5]{index=5}

---

## Limitations

- 대규모 학습 데이터가 필요하다.
- TCAD 데이터 생성 비용이 여전히 크다.
- 학습 범위를 벗어난 일반화에는 한계가 있다.
- Black-box 특성으로 물리적 해석이 어려울 수 있다.
- Physics Constraint가 없으면 신뢰성이 떨어질 수 있다.
- 최신 연구는 Physics-informed 접근을 함께 사용하는 방향으로 발전하고 있다. :contentReference[oaicite:6]{index=6}

---

## Key Takeaways

- Machine Learning은 반도체 Device Modeling의 핵심 기술로 자리 잡고 있다.
- 가장 큰 활용 분야는 TCAD Acceleration과 Surrogate Modeling이다.
- Compact Modeling과 Parameter Extraction에도 활발히 적용되고 있다.
- 최근에는 PINN, Neural Operator, Graph Neural Network 등 Physics-informed AI가 중요한 연구 방향이다.
- 미래의 Semiconductor Virtual R&D는 Physics와 AI의 결합이 중심이 될 가능성이 높다. :contentReference[oaicite:7]{index=7}

---

## Personal Notes

### Relation to My Current Project

현재 GitHub 프로젝트는

```
Physics-AI-Lab

↓

PINN

↓

DeepONet

↓

FNO

↓

MeshGraphNet

↓

Physics GNN

↓

TCAD Surrogate
```

순으로 진행하고 있다.

이 논문는 지금까지 공부한 모든 기술들이

반도체 산업에서 어디에 적용되는지를

하나의 로드맵으로 정리해준다.

---

### Relation to SK hynix TCAD

SK하이닉스 JD를 보면

- Material Simulation
- Physics AI
- PINN
- Neural Operator
- MLIP
- Device Simulation

등이 모두 포함되어 있다.

이 논문에서 소개하는 연구 방향과

거의 동일한 방향이라고 볼 수 있다.

즉

```
Material Simulation

↓

TCAD

↓

Machine Learning

↓

Physics AI

↓

Virtual R&D
```

가 앞으로의 핵심 기술 스택이다. :contentReference[oaicite:8]{index=8}

---

### Comparison with Previous Papers

| Paper | Main Contribution |
|---------|------------------------------|
| PINN | Physics Constraint Learning |
| DeepONet | Operator Learning |
| FNO | Fourier Operator |
| MeshGraphNet | Mesh-based Physics Solver |
| Physics GNN | Graph Physics |
| MLIP | Atomistic Potential Modeling |
| PhysicsNeMo | Unified Physics AI Framework |
| **Machine Learning for Semiconductor Device Modeling** | Semiconductor AI Roadmap & Survey |

---

### Why This Paper Matters

지금까지 리뷰한 논문들이 각각 하나의 알고리즘을 설명했다면,

이 논문은 **반도체 산업에서 Machine Learning이 실제 어떻게 활용되고 있는지 전체 지도를 제공하는 로드맵**이다.

특히 앞으로 공부해야 할

- Physics-informed AI
- TCAD Surrogate
- Compact Modeling
- Device Optimization
- Material AI

사이의 연결 관계를 이해하는 데 가장 좋은 출발점이다.

---

### Future Learning Plan

- [ ] TCAD Surrogate Model 구현
- [ ] Physics-informed Neural Operator 구현
- [ ] Graph-based Device Simulation 구현
- [ ] Compact Model Neural Network 구현
- [ ] Device Parameter Extraction AI 구현
- [ ] Material Simulation + MLIP 프로젝트 진행
- [ ] PhysicsNeMo 기반 Semiconductor AI 프로젝트 확장

---

### Paper Link

IEEE Access (2025)

A Comprehensive Review of Machine Learning Approaches for Semiconductor Device Modeling and Simulation

DOI: **10.1109/ACCESS.2025.3605856** :contentReference[oaicite:9]{index=9}