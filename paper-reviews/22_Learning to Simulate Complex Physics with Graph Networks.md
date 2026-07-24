# Paper Name

> Sanchez-Gonzalez et al. (2020)
>
> Learning to Simulate Complex Physics with Graph Networks
>
> International Conference on Machine Learning (ICML 2020)

---

## Why I Read This Paper

Graph Network-based Simulators(GNS)는 **MeshGraphNet의 직접적인 선행 연구**이며, Graph Neural Network(GNN)를 이용해 물리 시뮬레이션을 학습하는 대표적인 논문이다.

기존의 Physics-informed Neural Networks(PINNs)는 PDE를 직접 Loss에 포함시키고, Fourier Neural Operator(FNO)는 PDE 연산자를 학습한다면, GNS는 **물체를 그래프로 표현하여 시간에 따른 상태 변화를 직접 예측**한다.

반도체 TCAD에서도 Device Mesh를 Graph로 표현하여 전위(Potential), 전하(Carrier Density), Electric Field 등을 예측하는 연구가 활발히 진행되고 있어, GNS는 이러한 Graph 기반 Device Simulation의 기초가 되는 논문이다.

---

## Problem

기존의 물리 시뮬레이션은

- Finite Element Method (FEM)
- Finite Difference Method (FDM)
- Finite Volume Method (FVM)

등의 수치해석 기법을 사용한다.

이 방법들은 매우 정확하지만,

- 계산 시간이 길고
- Mesh가 커질수록 계산량이 급격히 증가하며
- 반복적인 Simulation Cost가 매우 크다.

또한 기존 Deep Learning은

- Grid 형태 입력에는 강하지만
- Mesh나 Particle처럼 불규칙한 구조를 처리하기 어렵다.

즉,

**복잡한 물리 시스템을 일반화하면서도 빠르게 시뮬레이션하는 모델이 필요했다.**

---

## Key Idea

모든 물리 시스템을

**Graph**

로 표현한다.

즉,

```
Particle

↓

Node

Interaction

↓

Edge

Simulation

↓

Message Passing
```

각 Node는

- 위치(Position)
- 속도(Velocity)
- 질량(Mass)
- 압력(Pressure)
- 기타 물리량

을 가진다.

Edge는

Node 간의

- 거리
- 힘
- 상호작용

을 의미한다.

Graph Neural Network는

Message Passing을 반복하면서

다음 시간(step)의 상태를 예측한다.

즉,

```
State(t)

↓

Graph Network

↓

State(t+1)
```

를 직접 학습한다.

---

## Method

### Step 1. Graph Construction

Simulation Domain을 Graph로 변환한다.

Node

- Particle
- Mesh Vertex

Edge

- Neighbor Relation
- Physical Interaction

각 Node Feature에는

- Position
- Velocity
- Material Property

등이 포함된다.

---

### Step 2. Encoder

Node Feature를 Latent Space로 변환한다.

\[
h_i^0=f_{enc}(x_i)
\]

여기서

- \(x_i\) : 입력 Feature
- \(h_i\) : Latent Representation

이다.

---

### Step 3. Message Passing

Node들은 주변 Node와 정보를 주고받는다.

Edge Message

\[
m_{ij}=f_e(h_i,h_j,e_{ij})
\]

Node Update

\[
h_i^{k+1}
=
f_v
\left(
h_i^k,
\sum_j m_{ij}
\right)
\]

이 과정을 여러 번 반복한다.

---

### Step 4. Decoder

최종 Latent Feature를

다음 시간 Step의 물리량으로 변환한다.

\[
y_i=f_{dec}(h_i)
\]

---

### Step 5. Rollout

한 Step 예측이 끝나면

이를 다시 입력으로 넣는다.

```
t

↓

t+1

↓

t+2

↓

...

↓

Future Simulation
```

이 방식으로 장시간 Simulation을 수행한다.

---

## Equation

Graph Update

\[
m_{ij}
=
f_e
(h_i,h_j,e_{ij})
\]

\[
h_i^{k+1}
=
f_v
\left(
h_i^k,
\sum_j m_{ij}
\right)
\]

Prediction

\[
x_{t+1}
=
G(x_t)
\]

Loss

\[
L
=
\sum_i
\|
x_i^{pred}
-
x_i^{true}
\|^2
\]

즉,

Physics PDE를 직접 푸는 대신

시간에 따른 State Transition을 학습한다.

---

## Advantages

- 불규칙한 Mesh 구조 처리 가능
- Particle System 표현이 자연스러움
- 다양한 물리 문제에 적용 가능
- FEM/FVM Solver보다 매우 빠른 추론 가능
- 다양한 Geometry에서도 일반화 성능 우수
- Message Passing을 통해 Local Physics를 효과적으로 학습
- MeshGraphNet의 기반이 되는 핵심 구조

---

## Limitations

- 매우 긴 Rollout에서는 오차가 누적될 수 있다.
- 학습 데이터가 충분하지 않으면 일반화가 어렵다.
- Global Constraint를 직접 보장하지 않는다.
- 매우 큰 Graph에서는 Message Passing 비용이 증가한다.
- PDE를 명시적으로 만족하는 것은 아니다.

---

## Key Takeaways

- Graph Network는 Physics Simulator를 학습할 수 있다.
- 물리 시스템을 Graph 형태로 표현하면 복잡한 Geometry도 처리할 수 있다.
- Message Passing을 반복하여 시간에 따른 상태 변화를 예측한다.
- MeshGraphNet은 GNS를 Mesh 기반으로 확장한 모델이다.
- 최근 Semiconductor Device Simulation에서도 Graph 기반 접근의 핵심 아이디어가 되고 있다.

---

## Personal Notes

### Relation to My Current Project

현재 GitHub에서는

```
PINN

↓

DeepONet

↓

FNO

↓

MeshGraphNet
```

순으로 학습하고 있다.

GNS는

MeshGraphNet 이전에 반드시 이해해야 하는

**Graph Physics의 출발점**이다.

---

### Relation to Semiconductor TCAD

반도체 TCAD에서는

Device Mesh를 그대로 Graph로 변환할 수 있다.

예를 들어

Node

- Potential
- Electron Density
- Hole Density
- Electric Field
- Temperature

Edge

- Mesh Connectivity
- Carrier Diffusion
- Electric Coupling

등으로 표현 가능하다.

Graph Network는

```
Mesh

↓

Graph

↓

Message Passing

↓

Potential Prediction
```

과 같은 방식으로

TCAD Solver를 Surrogate Model로 대체할 가능성을 보여준다.

---

### Comparison with Previous Papers

| Paper | Core Idea |
|--------|-----------|
| PINN | PDE Residual Learning |
| DeepONet | Operator Learning |
| FNO | Fourier Operator Learning |
| MeshGraphNet | Mesh 기반 Graph Simulation |
| **Graph Network-based Simulators** | Particle/Graph Dynamics Learning |

---

### Why This Paper Matters

이 논문은

**Graph Neural Network를 Physics Simulation에 본격적으로 적용한 최초의 대표 연구**이다.

오늘날

- MeshGraphNet
- Physics GNN
- Semiconductor Device GNN
- TCAD Graph Surrogate

등 대부분의 Graph 기반 Physics AI 연구는

이 논문의 Message Passing 구조를 기반으로 발전하였다.

즉,

Physics AI 분야에서

PINN이 "Physics Constraint"의 시작이라면,

GNS는 "Graph Physics Simulation"의 시작이라고 볼 수 있다.

---

### Future Learning Plan

- [ ] Graph Network 직접 구현
- [ ] Message Passing Network 구현
- [ ] Particle Simulation 예제 재현
- [ ] MeshGraphNet 논문와 구조 비교
- [ ] TCAD Mesh를 Graph로 변환하는 프로젝트 진행
- [ ] Graph 기반 Device Surrogate Model 구현

---

### Paper Link

https://arxiv.org/abs/2002.09405