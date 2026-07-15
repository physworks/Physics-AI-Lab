# Graph Neural Networks for Semiconductor Device Modeling

> Brahma et al. (2023)
>
> Accelerated Modelling of Interfaces for Electronic Devices using Graph Neural Networks
>
> arXiv

---

## Why I Read This Paper

기존 TCAD 기반 Device Simulation은 높은 정확도를 제공하지만 계산 비용이 매우 크다.

특히 최신 반도체는

- 비정질(Amorphous)
- 다결정(Polycrystalline)
- 복잡한 Material Interface
- 비정형(Unstructured) Mesh

를 포함하기 때문에 기존 Grid 기반 AI 모델은 일반화에 한계가 있다.

이 논문은 반도체 소자를 Graph 형태로 표현하고 Graph Neural Network(GNN)를 이용하여 원자 수준 물성과 소자 특성을 동시에 학습하는 새로운 접근을 제안한다.

---

## Problem

기존 TCAD AI 모델은 대부분

```
Regular Grid

↓

CNN / FNO
```

또는

```
Mesh

↓

Interpolation

↓

MLP
```

구조를 사용한다.

하지만 실제 Device는

- 다양한 Mesh 구조
- 복잡한 Material Interface
- 불규칙한 Geometry

를 가지므로

Grid 기반 모델의 일반화 성능이 제한된다.

---

## Key Idea

핵심 아이디어는

반도체 소자를 Graph로 표현하는 것이다.

```
Atom

↓

Node
```

```
Chemical Bond

↓

Edge
```

또는

```
Mesh Vertex

↓

Node
```

```
Neighbor Relation

↓

Edge
```

로 표현한 뒤

Graph Neural Network가

Node 간 Message Passing을 통해

물리 정보를 전달하며

Device 특성을 예측한다.

즉,

Geometry 자체를 Graph 구조로 학습한다.

---

## Method

### Step 1. Graph Construction

Device를 Graph로 변환한다.

Node

- Atomic Position
- Material Property
- Doping
- Potential

Edge

- Neighbor Relation
- Bond Length
- Interface Connection

등을 포함한다.

---

### Step 2. Message Passing

각 Node는

인접 Node 정보를 반복적으로 집계한다.

```
Neighbor Feature

↓

Aggregation

↓

Node Update
```

이를 여러 Layer 반복한다.

---

### Step 3. Global Representation

전체 Device Graph를 하나의 Embedding으로 변환한다.

이를 통해

- Current
- IV Curve
- Potential Distribution
- Density of States

등을 예측한다.

---

### Step 4. Physics-aware Learning

단순 Graph Embedding이 아니라

Material Interface

Atomic Interaction

Boundary Condition

등 물리 정보를 Feature에 포함하여 학습한다.

최근 연구에서는

Physics Constraint를 Loss Function에 함께 적용하기도 한다.

---

## Equation

Node Feature를

\[
h_i
\]

라고 하면

Message Passing은

\[
m_i=\sum_{j\in N(i)}\phi(h_i,h_j,e_{ij})
\]

이다.

Node Update는

\[
h_i'=f(h_i,m_i)
\]

으로 수행된다.

최종 Device Prediction은

\[
y=Readout(H)
\]

으로 계산된다.

Loss는 일반적으로

\[
L=MSE(y,\hat y)
\]

를 최소화한다.

Physics Constraint를 추가하여

Hybrid Loss를 사용하는 연구도 증가하고 있다. :contentReference[oaicite:3]{index=3}

---

## Advantages

- Unstructured Mesh 처리 가능
- Complex Geometry 표현 가능
- Material Interface 자연스럽게 표현
- Graph 크기 변화에 대응 가능
- TCAD Surrogate에 적합
- 다양한 Device 구조에 일반화 가능

---

## Limitations

- Graph 생성 과정이 복잡하다.
- 매우 큰 Device에서는 메모리 사용량이 증가한다.
- Message Passing Layer가 깊어질수록 Over-smoothing 문제가 발생할 수 있다.
- 대규모 학습 데이터가 필요하다.
- 실제 산업 적용 사례는 아직 제한적이다.

---

## Key Takeaways

- Device를 Grid가 아닌 Graph로 표현한다.
- Node와 Edge에 물리 정보를 직접 포함할 수 있다.
- Material Interface 표현 능력이 뛰어나다.
- 복잡한 TCAD Geometry에 적합하다.
- 차세대 Physics AI 기반 TCAD Surrogate의 핵심 기술 중 하나이다.

---

## Personal Notes

### Relation to My Current Project

현재 프로젝트는

```
GCA Equation

↓

PINN
```

기반이다.

향후에는

```
PINN

↓

Graph Neural Network

↓

TCAD Surrogate
```

순으로 확장하고 싶다.

---

### Relation to My Career

현재 업무에서도

```
공정 변수

↓

소자 물성

↓

AI 모델

↓

양산
```

이라는 관계를 모델링하였다.

GNN 역시

Node 간 물리적 관계를 학습한다는 점에서

기존 경험과 매우 유사한 접근이다.

---

### Relation to SK hynix TCAD

SK하이닉스 JD에서 강조한

- Material Simulation
- Physics AI
- Device Simulation
- Virtual R&D

와 매우 잘 연결된다.

특히

복잡한 Device Structure를

Graph Representation으로 표현하는 기술은

차세대 TCAD AI의 중요한 방향이다.

---

### Comparison with Previous Papers

| Paper | Representation | 특징 |
|---------|---------------|----------------------------|
| PINN | PDE | Physics Constraint |
| DeepONet | Operator | Function Learning |
| FNO | Fourier Grid | Fast PDE Solver |
| MeshGraphNet | Mesh Graph | Mesh Simulation |
| Universal Device Encoding | Device Graph | TCAD Encoding |
| **GNN for Device Modeling** | Physical Graph | Material & Device Modeling |

---

### Relation to TCAD

향후 Device Simulation은

```
Material

↓

MLIP

↓

Graph Representation

↓

Graph Neural Network

↓

TCAD Surrogate

↓

Virtual R&D
```

구조로 발전할 가능성이 높다.

특히 GNN은

복잡한 Geometry와 Material Interface를 표현하는 핵심 기술이 될 것으로 예상된다.

---

### Future Learning Plan

- [ ] PyTorch Geometric 학습
- [ ] Message Passing 구현
- [ ] Mesh → Graph 변환
- [ ] TCAD Mesh Graph 생성
- [ ] GNN 기반 IV Prediction
- [ ] GNN 기반 Potential Prediction
- [ ] PINN + GNN Hybrid Model 구현

---

### Paper Link

https://arxiv.org/abs/2310.06995