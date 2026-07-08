# Physics-informed AI Accelerated Retention Analysis of Ferroelectric Vertical NAND: From Day-Scale TCAD to Second-Scale Surrogate Model

> **Jinwoo Kim et al. (2025)**
>
> **Physics-informed AI Accelerated Retention Analysis of Ferroelectric Vertical NAND: From Day-Scale TCAD to Second-Scale Surrogate Model**
>
> **Preprint / Semiconductor AI Research**

---

## Why I Read This Paper

이 논문은 **Ferroelectric Vertical NAND(FE-VNAND)**의 Retention 특성을 분석하기 위해 기존 TCAD를 AI Surrogate Model로 대체한 연구입니다.

기존에는 Retention 특성을 분석하기 위해 수많은 TCAD Simulation을 반복 수행해야 했지만, 본 연구에서는 **Physics-informed AI**를 이용하여 Simulation 시간을 **수 일(Day)** 수준에서 **수 초(Second)** 수준으로 단축하였습니다.

SK하이닉스가 추진하고 있는 **AI Physics**, **Virtual R&D**, **TCAD Surrogate Model**과 매우 유사한 연구 방향이기 때문에 읽게 되었습니다.

또한 실제 반도체 소자 개발에서 Scientific Machine Learning이 어떻게 활용되는지 이해하고, 향후 제 프로젝트 방향을 설정하기 위해 리뷰하였습니다.

---

## Problem

차세대 Vertical NAND에서는 Charge Trap Flash 대신 **Ferroelectric Memory**를 적용하려는 연구가 활발히 진행되고 있습니다.

하지만 Ferroelectric Memory는 시간이 지나면서 저장된 Polarization이 감소하는 **Retention Degradation** 문제가 존재합니다.

Retention을 정확하게 예측하기 위해서는

- Polarization Switching
- Trap Dynamics
- Electric Field Distribution
- Time-dependent Charge Transport

등 다양한 물리 현상을 동시에 계산해야 합니다.

기존 TCAD에서는

- 하나의 Simulation에도 수 시간 이상이 필요하고
- 다양한 공정 변수와 구조를 탐색하기 위해서는 수천 번 이상의 Simulation이 요구됩니다.

결과적으로 전체 연구에는 수 일 이상의 계산 시간이 필요하여 실제 연구개발 속도를 크게 제한하는 문제가 있습니다.

---

## Key Idea

핵심 아이디어는

**Physics-informed AI Surrogate Model**을 이용하여 TCAD Simulation 결과를 학습시키는 것입니다.

```
Device Structure
+

Material Parameters
+

Bias Condition

↓

Physics-informed AI

↓

Retention Prediction
```

AI 모델은

- Device Geometry
- Material Parameter
- Operation Condition

을 입력받아

Retention Characteristic을 직접 예측합니다.

Physics Constraint를 함께 고려하기 때문에 단순 Data-driven Model보다 높은 일반화 성능을 확보할 수 있습니다.

결과적으로 기존 TCAD를 반복 수행하지 않고도 Retention을 매우 빠르게 예측할 수 있습니다.

---

## Method

전체 과정은 다음과 같습니다.

### Step 1. TCAD Dataset Generation

다양한

- Channel Diameter
- Oxide Thickness
- Ferroelectric Thickness
- Temperature
- Bias Condition

에 대해 TCAD Simulation을 수행합니다.

이를 통해 학습용 Dataset을 생성합니다.

---

### Step 2. Physics-informed AI Training

생성된 TCAD 결과를 이용하여 AI 모델을 학습합니다.

Loss Function은

- Data Loss
- Physics Constraint

를 동시에 고려합니다.

즉,

```
Loss

=

Prediction Error

+

Physics Constraint
```

형태입니다.

---

### Step 3. Surrogate Inference

학습 완료 후에는

새로운 Device Parameter가 입력되면

AI가 즉시 Retention 특성을 예측합니다.

기존 TCAD를 수행할 필요가 없습니다.

---

## Equation

논문의 핵심 개념은 다음과 같이 표현할 수 있습니다.

\[
f_{\theta}
:
(\text{Geometry},
\text{Material},
\text{Bias},
t)
\rightarrow
Retention
\]

여기서

- Geometry
- Material Parameter
- Bias Condition
- Time

을 입력받아

Retention Characteristic을 예측합니다.

Physics-informed Learning을 적용하여

Prediction이 실제 물리 거동과 일치하도록 학습합니다.

---

## Advantages

- Day 수준의 TCAD Simulation을 Second 수준으로 단축
- Device Optimization 속도 향상
- Virtual R&D 적용 가능
- Physics Constraint를 이용하여 일반화 성능 향상
- Design Space Exploration 가능
- 반도체 Memory 개발 기간 단축

---

## Limitations

- 초기 TCAD Dataset 생성 비용이 큼
- 새로운 Device Structure에서는 재학습이 필요할 수 있음
- Training Data 품질이 모델 성능을 결정함
- Physics Constraint 설계가 어려움
- 실제 양산 환경에서는 지속적인 Validation이 필요함

---

## Key Takeaways

- TCAD는 AI Surrogate Model로 매우 빠르게 대체될 수 있다.
- Scientific Machine Learning은 반도체 Device Simulation의 핵심 기술이 되고 있다.
- Physics-informed AI는 단순 Deep Learning보다 높은 신뢰성을 제공한다.
- Surrogate Model은 차세대 Virtual R&D의 핵심 구성 요소이다.
- 반도체 연구개발은 AI 기반 Design Space Exploration으로 빠르게 변화하고 있다.

---

## Personal Notes

### Relation to My Current Project

현재 GitHub에서 진행 중인 TFT GCA 기반 PINN 프로젝트는 비교적 단순한 Device Physics를 대상으로 합니다.

반면 이 논문는 실제 산업에서

- TCAD
- Physics-informed AI
- Surrogate Model

을 결합하여 Device Simulation을 가속화한 사례입니다.

현재 프로젝트의 최종 목표 역시

```
Physics Equation

↓

PINN

↓

TCAD Surrogate

↓

Virtual R&D
```

이므로 매우 좋은 참고 사례가 되었습니다.

---

### Relation to My Career

제가 수행했던 업무 역시

```
Physics 분석

↓

모델링

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

TCAD

↓

Physics-informed AI

↓

Surrogate Model

↓

Virtual R&D
```

라는 동일한 철학을 보여줍니다.

적용 분야는 다르지만,

물리 기반 모델을 AI로 근사하여 연구개발 효율을 높인다는 점에서 제 커리어 방향과 매우 유사하다고 느꼈습니다.

---

### Future Learning Plan

이 논문를 바탕으로 앞으로 다음 프로젝트를 진행할 계획입니다.

- [ ] PINN 기반 TCAD Surrogate 구현
- [ ] Retention Prediction Toy Project 제작
- [ ] Neural Operator 적용
- [ ] PhysicsNeMo 실습
- [ ] Process TCAD Surrogate Model 구현
- [ ] Memory Device Scientific ML 프로젝트 확장

---

### Paper Link
https://arxiv.org/pdf/2603.06881