# Using AI Physics for Technology Computer-Aided Design Simulations

> **Ram Cherukuri, Kihang Youn, Gyuseung Han, Min Kang, Hwiwon Seo, Junghan Kim et al. (2025)**
>
> **Using AI Physics for Technology Computer-Aided Design Simulations**
>
> **NVIDIA Technical Blog / SK hynix Research**

---

## Why I Read This Paper

이 글은 NVIDIA와 SK하이닉스 TCAD Intelligence Team이 공동으로 소개한 **AI Physics 기반 TCAD 가속화 전략**을 설명하는 기술 문서입니다.

기존 논문들이 특정 AI 모델(PINN, DeepONet 등)의 알고리즘을 제안하는 데 초점을 맞추었다면, 이 글은 AI를 활용하여 TCAD 연구개발 방식을 어떻게 변화시키고 있는지 산업 관점에서 설명합니다.

특히 SK하이닉스가 앞으로 추진하는 **Virtual R&D**와 **Scientific Machine Learning**의 방향을 이해하고, 차세대 TCAD 엔지니어에게 요구되는 핵심 기술이 무엇인지 파악하기 위해 읽었습니다.

---

## Problem

TCAD(Technology Computer-Aided Design)는 반도체 공정과 소자의 물리적 거동을 시뮬레이션하여 실제 제조 이전에 다양한 조건을 검증하는 핵심 기술입니다.

대표적인 활용 분야는 다음과 같습니다.

- Process TCAD
- Device TCAD
- 전기적 특성 예측
- 공정 최적화

하지만 최신 반도체 공정에서는 고려해야 하는 설계 변수와 공정 조건이 매우 많아졌습니다.

예를 들어,

- 식각 조건
- 증착 조건
- Material 조성
- Device Geometry
- Process Variation

등을 모두 고려하면 수천~수만 번의 TCAD Simulation이 필요합니다.

기존 TCAD는 하나의 Simulation에도 수 시간에서 수 일이 소요되므로 전체 설계 공간을 탐색하는 것이 현실적으로 매우 어렵습니다.

즉, 현재의 가장 큰 병목은 **Simulation 정확도보다 Simulation 속도**입니다.

---

## Key Idea

핵심 아이디어는 기존 TCAD Simulation을 반복 수행하는 대신,

**AI Physics 기반 Surrogate Model**을 학습시키는 것입니다.

```
Material / Process Parameters
            │
            ▼
      AI Surrogate Model
            │
            ▼
 Predicted TCAD Result
```

AI 모델은 기존 TCAD Simulation 데이터를 학습하여

공정 조건 → Simulation 결과

사이의 관계를 직접 예측합니다.

학습이 완료되면 기존에는 수 시간 또는 수 일이 걸리던 Simulation을 **수 ms 수준**으로 수행할 수 있으며, 이를 통해 Virtual R&D 환경에서 대규모 Design Space Exploration이 가능해집니다.

---

## Method

이 글은 하나의 AI 모델을 제안하기보다 NVIDIA PhysicsNeMo를 기반으로 한 AI Physics Framework를 소개합니다.

### PhysicsNeMo

Scientific Machine Learning을 위한 NVIDIA Framework입니다.

지원하는 주요 모델은 다음과 같습니다.

- Physics-Informed Neural Networks (PINNs)
- Neural Operators
- Graph Neural Networks (GNN)
- Transformer

---

### AI Surrogate Models

기존처럼 PDE를 반복적으로 계산하는 대신,

```
Simulation Input

↓

AI Model

↓

Simulation Output
```

형태의 AI 모델을 학습하여 매우 빠른 추론 속도를 제공합니다.

---

### Graph Neural Networks

식각(Process TCAD)과 같이 Mesh 구조가 계속 변화하는 문제에서는

다음과 같은 GNN 기반 모델을 활용합니다.

- Graph Network Simulator (GNS)
- MeshGraphNet (MGN)

Mesh 기반 정보를 직접 활용하기 때문에 공정 시뮬레이션에 적합합니다.

---

### Engineering Improvements

SK하이닉스 TCAD Intelligence Team은 다음과 같은 기법을 추가하여 모델 성능을 향상시켰습니다.

- MeshGraphNet 기반 Memory 최적화
- Chamfer Loss 적용
- Re-meshing
- Feature Selection 개선
- Multi-scale Message Passing
- Dynamic Material Feature Update

이를 통해 Process TCAD의 정확도와 안정성을 향상시켰습니다.

---

## Equation

이 글에서는 특정 물리 방정식을 제안하지는 않습니다.

대신 전체 문제를 다음과 같은 함수 근사 문제로 정의합니다.

\[
f_{\theta}:
(\text{Process Parameters},
\text{Material Properties})
\rightarrow
\text{TCAD Simulation Output}
\]

즉,

공정 조건과 Material 정보를 입력받아

기존 TCAD Simulation 결과를 직접 예측하는 AI 모델을 구축하는 것이 목표입니다.

---

## Advantages

- 기존 TCAD 대비 매우 빠른 Simulation 수행
- 대규모 Virtual R&D 가능
- 다양한 Scientific ML 모델(PINN, Neural Operator, GNN 등) 활용 가능
- 반도체 공정 최적화 비용 절감
- GPU 기반 대규모 병렬 연산 지원
- Design Space Exploration 효율 향상

---

## Limitations

- 많은 양의 TCAD Simulation 데이터가 필요함
- 학습 데이터의 품질이 모델 성능을 결정함
- 학습하지 않은 조건에 대한 일반화에는 한계가 존재함
- 기존 Physics 기반 TCAD를 완전히 대체할 수는 없음
- 실제 TCAD와의 지속적인 검증이 필요함

---

## Key Takeaways

- AI Physics는 차세대 반도체 Virtual R&D의 핵심 기술이다.
- TCAD는 단순 Simulation Tool에서 AI 기반 최적화 플랫폼으로 발전하고 있다.
- Surrogate Model을 이용하면 Simulation 시간을 획기적으로 단축할 수 있다.
- Process TCAD에서는 Graph Neural Network가 매우 중요한 역할을 한다.
- NVIDIA PhysicsNeMo는 PINN, Neural Operator, GNN 등을 통합한 Scientific Machine Learning Framework이다.
- 앞으로 반도체 연구개발은 Physics와 AI를 함께 활용하는 방향으로 발전할 것이다.

---

## Personal Notes

### Relation to My Current Project

현재 GitHub에서는 TFT Id-Vg 특성을 대상으로 GCA 기반 PINN 프로젝트를 진행하고 있습니다.

이번 글을 통해 PINN뿐 아니라 실제 산업에서는

- Neural Operator
- Graph Neural Network
- AI Surrogate Model

등 다양한 Scientific Machine Learning 기술을 함께 활용한다는 점을 이해할 수 있었습니다.

---

### Relation to My Career

제가 지금까지 수행한 업무는

```
Physics 분석

↓

모델링

↓

Machine Learning

↓

양산 시스템 적용
```

의 흐름이었습니다.

반면 SK하이닉스가 제시하는 AI Physics Framework는

```
Device Physics

↓

TCAD Simulation

↓

AI Surrogate Model

↓

Virtual R&D
```

의 구조입니다.

비록 적용 대상은 다르지만,

"물리 현상을 이해하고 모델링하여 실제 연구개발에 활용한다"는 본질은 동일하다고 생각합니다.

기존 디스플레이에서 쌓은 Physics 기반 알고리즘 개발 경험을 반도체 Device Simulation 분야로 확장할 수 있다는 확신을 갖게 되었습니다.

---

### Future Learning Plan

향후 다음 순서로 프로젝트를 확장할 계획입니다.

- [ ] DeepONet 구현
- [ ] Fourier Neural Operator 구현
- [ ] MeshGraphNet 학습 및 구현
- [ ] NVIDIA PhysicsNeMo 실습
- [ ] AI 기반 TCAD Surrogate Model 구현
- [ ] Semiconductor Physics AI 프로젝트 확장

장기적으로는 AI Physics 기반 Semiconductor Virtual R&D를 구현할 수 있는 엔지니어를 목표로 학습을 이어갈 예정입니다.

---

### Paper Link

- SK hynix Research
  https://research.skhynix.com/blog/detail?seq=229

- NVIDIA Technical Blog
  https://developer.nvidia.com/blog/using-ai-physics-for-technology-computer-aided-design-simulations/