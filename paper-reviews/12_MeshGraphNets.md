# MeshGraphNets

> Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, Peter Battaglia, Peter W. Battaglia, et al. (2021)
>
> **Learning Mesh-Based Simulation with Graph Networks**
>
> International Conference on Learning Representations (ICLR 2021)

---

## Why I Read This Paper

기존 Scientific Machine Learning은 주로 격자(Grid) 기반 데이터나 정형화된 PDE 문제를 대상으로 발전해 왔습니다.

하지만 실제 반도체 TCAD, 유한요소해석(FEM), 구조해석, 유체해석 등 대부분의 산업용 시뮬레이션은 **비정형 Mesh(Unstructured Mesh)** 위에서 수행됩니다.

MeshGraphNets(MGN)는 이러한 Mesh를 Graph로 표현하고 Graph Neural Network(GNN)를 이용해 물리 시뮬레이션 자체를 학습하는 최초의 대표적인 연구입니다.

특히 TCAD 역시 Device Geometry를 Mesh로 분할하여 Poisson Equation, Carrier Continuity Equation 등을 계산하기 때문에, 향후 AI 기반 TCAD Simulation을 이해하기 위해 반드시 읽어야 할 논문이라고 판단했습니다.

---

## Problem

기존 Numerical Simulation은

- Finite Element Method (FEM)
- Finite Volume Method (FVM)
- Finite Difference Method (FDM)

등을 이용하여 PDE를 반복적으로 계산합니다.

하지만

- Mesh가 촘촘할수록 계산량이 증가
- 복잡한 Geometry일수록 Solver가 느려짐
- Parameter Sweep에 막대한 시간이 필요

합니다.

기존 CNN 기반 AI 모델은

```
Regular Grid
```

에서는 잘 동작하지만

```
Irregular Mesh
```

에서는 적용이 어렵습니다.

즉,

실제 산업용 Simulation에는 활용이 제한됩니다.

---

## Key Idea

논문의 핵심은

**Mesh 자체를 Graph로 변환하여 Physics를 학습하는 것**입니다.

기존 방식

```
Mesh

↓

FEM Solver

↓

Next State
```

제안 방식

```
Mesh

↓

Graph Neural Network

↓

Next State
```

Mesh의

- Node
- Edge
- Connectivity

를 그대로 유지한 채

Graph Message Passing으로

물리 현상을 예측합니다.

즉,

PDE를 직접 푸는 것이 아니라

Mesh 위에서 발생하는 Physics Evolution을 학습합니다.

---

## Method

### Step 1. Mesh를 Graph로 변환

Finite Element Mesh

↓

Graph

각 Node는

- Position
- Velocity
- Pressure
- Temperature
- Material Property

등의 정보를 가집니다.

Edge는

Node 간 연결 관계를 표현합니다.

---

### Step 2. Message Passing

각 Node는

인접 Node로부터 정보를 전달받습니다.

```
Neighbor

↓

Message

↓

Aggregation

↓

Node Update
```

이를 여러 번 반복하면서

Physics Information이

전체 Mesh로 전달됩니다.

---

### Step 3. Physics Prediction

최종적으로

각 Node의

- Velocity
- Pressure
- Position
- Stress

등을 예측합니다.

Prediction 결과를

다음 Time Step의 입력으로 사용하여

Simulation을 수행합니다.

---

### Step 4. Rollout

한 Step만 예측하는 것이 아니라

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

까지 반복적으로 Simulation을 진행합니다.

이를 Rollout이라 합니다.

---

## Equation

Graph는

\[
G=(V,E)
\]

로 표현됩니다.

여기서

- \(V\) : Node
- \(E\) : Edge

입니다.

Message Passing은

\[
m_{ij}
=
\phi_e(h_i,h_j,e_{ij})
\]

Node Update는

\[
h_i'
=
\phi_v
\left(
h_i,
\sum_j m_{ij}
\right)
\]

으로 수행됩니다.

즉,

이웃 Node들의 정보를 모두 모아

새로운 Node State를 계산합니다.

Loss는

\[
L
=
MSE(y,\hat y)
\]

를 사용하여

Simulation 결과와 비교합니다.

---

## Advantages

- Irregular Mesh 처리 가능
- FEM/FVM Mesh 그대로 사용 가능
- 다양한 Geometry에 일반화 가능
- 기존 Numerical Solver보다 매우 빠른 추론
- Physics Simulation에 높은 정확도
- 실제 산업 문제에 적용 가능
- TCAD, CFD, Structural Mechanics 등에 활용 가능

---

## Limitations

- 대량의 Simulation Dataset이 필요하다.
- Rollout이 길어질수록 오차가 누적된다.
- 매우 복잡한 Multi-Physics에서는 일반화가 쉽지 않다.
- 학습 데이터 범위를 크게 벗어나면 정확도가 감소한다.
- Mesh 품질에 따라 성능 차이가 발생할 수 있다.

---

## Key Takeaways

- Mesh를 Graph로 표현하면 복잡한 Geometry도 AI가 처리할 수 있다.
- Graph Neural Network는 Physics Simulation과 매우 잘 맞는다.
- MeshGraphNet은 산업용 Simulation AI의 대표적인 구조이다.
- 향후 TCAD Surrogate Model에서도 GNN 활용 가능성이 매우 높다.
- Operator Learning 이후 가장 중요한 Simulation AI 기술 중 하나이다.

---

## Personal Notes

### Relation to My Current Project

현재 진행 중인 프로젝트는

```
Physics Equation

↓

PINN

↓

Id-Vg Prediction
```

이다.

하지만

실제 TCAD는

```
Device Geometry

↓

Mesh

↓

Poisson Solver

↓

Carrier Transport
```

를 수행한다.

MeshGraphNet은

이 과정을

Graph Neural Network로

대체하는 접근이다.

---

### Relation to My Career

MeshGraphNet 역시

Geometry

↓

Physics

↓

AI

↓

Simulation

이라는 동일한 철학을 가진다.

---

### Relation to SK hynix TCAD

SK하이닉스 JD에서 강조한

- Scientific Machine Learning
- Physics AI
- Material Simulation
- Virtual R&D

를 실제 구현할 수 있는 핵심 기술 중 하나이다.

향후 TCAD에서는

```
TCAD Mesh

↓

MeshGraphNet

↓

Instant Device Simulation
```

형태의 연구가 활발해질 가능성이 높다.

---

### Comparison with Previous Papers

| Paper | 학습 대상 | 입력 형태 | Physics 활용 |
|---------|-------------------------------|----------------|----------------|
| PINN | PDE Solution | Point | 매우 높음 |
| DeepONet | Operator | Function | 높음 |
| FNO | Fourier Operator | Grid | 높음 |
| Neural Operator | General Operator | Function | 높음 |
| MeshGraphNet | Mesh Simulation | Graph | 매우 높음 |

MeshGraphNet은

Operator Learning을

비정형 Mesh까지 확장한 대표적인 연구이다.

---

### Relation to TCAD

TCAD에서는

- Device Geometry 생성
- Mesh Generation
- Poisson Equation
- Electron/Hole Continuity Equation
- Heat Equation

등을 Mesh 위에서 계산한다.

따라서

MeshGraphNet은

향후

- AI TCAD
- Surrogate TCAD
- Device Physics AI

에서 가장 활용 가능성이 높은 모델 중 하나이다.

---

### Future Learning Plan

이 논문를 기반으로 다음 프로젝트를 진행할 계획이다.

- [ ] PyTorch Geometric 학습
- [ ] Graph Neural Network 구현
- [ ] MeshGraphNet 구현
- [ ] 간단한 FEM Dataset 생성
- [ ] TCAD Mesh Dataset 적용
- [ ] MeshGraphNet 기반 TCAD Surrogate 실험

---

### Paper Link

https://arxiv.org/abs/2010.03409