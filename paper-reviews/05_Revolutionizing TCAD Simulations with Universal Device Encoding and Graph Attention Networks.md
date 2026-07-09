# Revolutionizing TCAD Simulations with Universal Device Encoding and Graph Attention Networks

> **Yash Agrawal et al. (2023)**
>
> **Revolutionizing TCAD Simulations with Universal Device Encoding and Graph Attention Networks**
>
> **arXiv Preprint**

---

## Why I Read This Paper

기존 TCAD는 높은 정확도를 제공하지만, 하나의 Device를 해석하는 데에도 많은 계산 시간이 필요합니다. 특히 공정 변수, 구조 변수, Bias 조건을 동시에 탐색하는 Design Space Exploration에서는 수천~수만 번의 TCAD Simulation이 필요하여 연구개발 속도를 제한하는 요인이 됩니다.

이 논문은 **Graph Neural Network(GNN)** 기반 AI를 이용하여 Device 내부의 물리 정보를 직접 학습하고, 기존 TCAD Solver를 매우 빠르게 대체하는 방법을 제안합니다.

특히 Device를 단순한 이미지나 벡터가 아닌 **Graph 구조**로 표현하여 다양한 반도체 구조에 일반화할 수 있다는 점이 매우 인상적이었습니다.

SK하이닉스가 추진하는 AI Physics 및 Virtual R&D 방향과도 매우 유사하며, 향후 TCAD Surrogate Model이 어떤 형태로 발전할 수 있는지를 이해하기 위해 읽게 되었습니다.

---

## Problem

기존 TCAD Simulation은 다음과 같은 문제를 가지고 있습니다.

- Poisson Equation
- Drift-Diffusion Equation
- Carrier Continuity Equation

등의 PDE(Partial Differential Equation)를 반복적으로 계산해야 합니다.

Device가 복잡해질수록

- Mesh 수 증가
- Newton Iteration 증가
- Sparse Matrix 계산 증가

가 동시에 발생합니다.

결과적으로

- Simulation 시간이 길고
- Parameter Sweep가 매우 비효율적이며
- AI 기반 Device Optimization 적용이 어렵습니다.

또한 기존 CNN 기반 AI는 Device 구조가 조금만 변경되어도 다시 학습해야 하는 문제가 존재했습니다.

---

## Key Idea

핵심 아이디어는

**Device 자체를 Graph로 표현하는 것**입니다.

즉,

기존 방식

```
Device

↓

Image

↓

CNN
```

이 아니라

```
Device Mesh

↓

Graph

↓

Graph Attention Network

↓

Potential Distribution
Carrier Density
Electric Field

↓

IV Prediction
```

을 사용합니다.

Graph는 Mesh 구조 자체를 그대로 표현하기 때문에

- Device Shape 변화
- Mesh 변화
- 구조 변경

에 훨씬 강한 일반화 성능을 가집니다.

---

## Method

전체 과정은 다음과 같습니다.

### Step 1. Universal Device Encoding

TCAD Mesh를 Graph 형태로 변환합니다.

각 Node는

- Position
- Material
- Doping
- Region
- Boundary Condition

등을 Feature로 가집니다.

Edge는

Mesh Connectivity를 그대로 유지합니다.

즉,

Device 전체가 하나의 Graph가 됩니다.

---

### Step 2. Graph Attention Network

Graph Attention Layer를 이용하여

각 Node가

주변 Node의 정보를 Attention Weight로 학습합니다.

이를 통해

- Potential Distribution
- Carrier Distribution

을 동시에 예측합니다.

Attention 구조 덕분에

Device 내부에서 중요한 영역을 자동으로 학습할 수 있습니다.

---

### Step 3. Physics Prediction

학습이 완료되면

새로운 Device가 입력되어도

Graph를 생성한 후

즉시

- Electric Potential
- Electron Density
- Hole Density

를 예측할 수 있습니다.

---

### Step 4. Device Characteristics

예측된 Potential과 Carrier Distribution으로부터

최종적으로

- IV Curve
- Device Characteristics

를 계산합니다.

TCAD Solver를 반복 수행할 필요가 없습니다.

---

## Equation

Graph는 다음과 같이 정의됩니다.

\[
G=(V,E)
\]

여기서

- V : Mesh Node
- E : Mesh Connectivity

각 Node Feature는

\[
x_i=
(Position,
Material,
Doping,
Boundary)
\]

형태를 가집니다.

Graph Attention은

\[
\alpha_{ij}
=
softmax
(a(Wx_i,Wx_j))
\]

으로 계산됩니다.

최종적으로

\[
Node_i
=
\sum_j
\alpha_{ij}
Wx_j
\]

를 이용하여

각 Node의 물리량을 예측합니다.

---

## Advantages

- Device Shape 변화에 강함
- Mesh 변경에도 일반화 가능
- CNN보다 높은 구조 확장성
- TCAD Solver 대비 매우 빠른 추론 속도
- 다양한 Device 구조에 동일 모델 적용 가능
- Virtual R&D에 적합
- Design Space Exploration 가속

---

## Limitations

- 초기 TCAD Dataset 구축 비용이 큼
- Graph 생성 과정이 필요함
- 매우 새로운 Device에서는 추가 학습이 필요
- Physics Constraint가 PINN만큼 직접적으로 포함되지는 않음
- 학습 데이터 품질에 성능이 크게 좌우됨

---

## Key Takeaways

- Graph는 Device Mesh를 표현하기에 가장 자연스러운 데이터 구조이다.
- Graph Neural Network는 CNN보다 Device 일반화 능력이 뛰어나다.
- Universal Device Encoding은 다양한 반도체 구조를 하나의 AI 모델로 다룰 수 있게 한다.
- GNN 기반 TCAD는 차세대 AI Physics의 핵심 기술 중 하나가 될 가능성이 높다.
- Scientific Machine Learning은 단순 회귀 모델을 넘어 Device Physics 자체를 학습하는 방향으로 발전하고 있다.

---

## Personal Notes

### Relation to My Current Project

현재 진행 중인 GCA 기반 PINN 프로젝트는 물리 방정식을 직접 Loss Function에 반영하여 TFT 특성을 학습합니다.

반면 이 논문는 Device 자체를 Graph 형태로 변환하여

```
Device Mesh

↓

Graph

↓

GNN

↓

Physics Prediction
```

을 수행합니다.

두 접근 모두 Physics 기반 AI이지만,

현재 프로젝트보다 훨씬 실제 TCAD 환경에 가까운 접근입니다.

향후 GitHub 프로젝트를

```
PINN

↓

Neural Operator

↓

Graph Neural Network

↓

TCAD Surrogate
```

순으로 확장하면 매우 자연스러운 로드맵이 될 것이라고 생각했습니다.

---

### Relation to My Career

제가 수행한 업무 역시

```
Physics 분석

↓

비선형 모델링

↓

Machine Learning

↓

양산 적용
```

이라는 흐름이었습니다.

이 논문에서는

```
Device Physics

↓

Graph Representation

↓

Graph AI

↓

Virtual TCAD
```

라는 동일한 철학을 보여줍니다.

특히 Device를 단순한 데이터가 아닌 **물리 구조 자체로 표현**한다는 점이 매우 인상 깊었습니다.

이는 앞으로 반도체 AI 연구가 단순 Regression이 아닌 Physics-aware Representation Learning으로 발전하고 있음을 보여주는 사례라고 생각합니다.

---

### Future Learning Plan

이 논문를 기반으로 앞으로 다음 프로젝트를 진행할 계획입니다.

- [ ] Graph Neural Network(GNN) 기본 이론 학습
- [ ] PyTorch Geometric 실습
- [ ] Device Mesh를 Graph로 변환하는 예제 구현
- [ ] GNN 기반 Potential Prediction 프로젝트
- [ ] TCAD Graph Surrogate Model 구현
- [ ] PhysicsNeMo Graph Module 학습

---

### Paper Link
https://arxiv.org/abs/2308.11624