# Automated, Physics-Guided AI Framework for Asymmetry-Aware Ferroelectric Compact Models

> Yansen Liu et al. (2025)
>
> **Automated, Physics-Guided AI Framework for Asymmetry-Aware Ferroelectric Compact Models**
>
> IEEE Transactions on Electron Devices (TED)

---

## Why I Read This Paper

최근 반도체 산업에서는 **Ferroelectric Memory(FeRAM, FeFET, FeCAP)** 가 차세대 비휘발성 메모리 후보로 주목받고 있다.

하지만 Ferroelectric 소자는 일반 MOSFET과 달리

- Hysteresis
- Polarization Switching
- Wake-up Effect
- Fatigue
- Imprint
- Asymmetry

등 매우 복잡한 물리 현상을 동시에 고려해야 한다.

기존 Compact Model은 이러한 현상을 사람이 직접 수식으로 모델링해야 하기 때문에 개발 시간이 길고 새로운 공정이 등장할 때마다 모델을 다시 작성해야 하는 문제가 있다.

이 논문은 **Physics와 AI를 결합하여 Compact Model을 자동으로 생성하는 Framework**를 제안한다.

특히 "AI가 Physics를 학습하는 것이 아니라 Physics가 AI를 안내(Guided Learning)한다."는 철학이 매우 인상적이었으며, SK하이닉스가 추진하는 AI 기반 Virtual R&D와 가장 가까운 연구 중 하나라고 판단하여 읽게 되었다.

---

## Problem

Ferroelectric Device는 일반 MOSFET보다 훨씬 복잡하다.

대표적으로

- Polarization Switching
- Domain Nucleation
- Domain Growth
- Internal Electric Field
- Charge Trapping

등이 동시에 발생한다.

기존 Compact Model 개발 과정은

```
TCAD

↓

Device Measurement

↓

Physics Analysis

↓

Hand-crafted Equation

↓

Verilog-A

↓

SPICE
```

순으로 진행된다.

이 과정은

- 개발 기간이 매우 길고
- 전문가 의존성이 높으며
- 새로운 공정마다 다시 모델을 작성해야 한다.

또한 Ferroelectric 소자의 비대칭성(Asymmetry)을 정확하게 표현하기 어렵다.

---

## Key Idea

논문의 핵심은

**Physics Constraint를 이용하여 AI가 Compact Model을 자동 생성하도록 만드는 것이다.**

기존 방식

```
Physics

↓

Manual Equation

↓

Compact Model
```

제안한 방식

```
Measurement

+

Physics Prior

↓

AI

↓

Compact Model
```

즉,

AI가 자유롭게 학습하는 것이 아니라

Physics가 학습 방향을 제한(Guided)한다.

이를 통해

- 물리적으로 일관된 결과
- 높은 일반화 성능
- 자동화된 Model Generation

을 동시에 달성한다.

---

## Method

### Step 1. Device Data 확보

Ferroelectric Device의

- P-V Curve
- I-V Curve
- Switching Characteristics

를 측정한다.

또는 TCAD Simulation으로 생성할 수도 있다.

---

### Step 2. Physics Prior 정의

AI가 반드시 만족해야 하는

물리적 특성을 정의한다.

예를 들어

- Polarization Saturation
- Switching Threshold
- Symmetry / Asymmetry
- Energy Stability

등을 Constraint로 사용한다.

---

### Step 3. AI Model 학습

Neural Network는

입력

- Voltage
- Time
- Temperature

출력

- Polarization
- Current

를 학습한다.

Loss는

```
Data Loss

+

Physics Loss
```

로 구성된다.

Physics Constraint 덕분에

학습 데이터가 부족한 영역에서도

물리적으로 타당한 결과를 출력한다.

---

### Step 4. Compact Model 생성

학습된 AI 모델은

자동으로

Compact Model 형태로 변환된다.

이를

SPICE Simulation

또는

Circuit Design

에 그대로 사용할 수 있다.

즉,

Device Characterization부터

Compact Model 생성까지

자동화하였다.

---

## Equation

Loss Function은

\[
L
=
L_{Data}
+
\lambda L_{Physics}
\]

이다.

Data Loss는

측정 데이터와 AI 출력의 오차이다.

\[
L_{Data}
=
MSE(y,\hat y)
\]

Physics Loss는

Ferroelectric Polarization의

물리적 특성을 만족하도록 구성된다.

예를 들어

- Polarization Saturation
- Switching Behavior
- Energy Constraint

등이 포함된다.

전체 Loss를 최소화하도록

Neural Network를 학습한다.

---

## Advantages

- Compact Model 생성 자동화
- Physics를 유지하면서 AI 활용 가능
- 새로운 공정에도 빠르게 적용 가능
- Ferroelectric Asymmetry를 효과적으로 표현
- 전문가 의존성 감소
- Model Development 기간 단축
- Virtual R&D Workflow와 자연스럽게 연결 가능

---

## Limitations

- Physics Constraint 설계가 쉽지 않다.
- 충분한 Device Characterization 데이터가 필요하다.
- 매우 복잡한 Ferroelectric Physics를 완전히 표현하기는 어렵다.
- 기존 Compact Model보다 해석성이 낮을 수 있다.
- 다른 Device로 확장하려면 Physics Constraint를 새롭게 정의해야 한다.

---

## Key Takeaways

- AI는 Physics를 대체하는 것이 아니라 Physics를 보완한다.
- Compact Model 개발도 AI로 자동화할 수 있다.
- Physics Prior를 활용하면 일반화 성능이 크게 향상된다.
- AI 기반 Compact Modeling은 앞으로 EDA의 중요한 연구 분야가 될 가능성이 높다.
- Device Simulation → Compact Model → Circuit Design 전체 Workflow가 AI 기반으로 자동화될 수 있다.

---

## Personal Notes

### Relation to My Current Project

진행 했었던

**TFT GCA 기반 PINN 프로젝트**는

```
Physics Equation

↓

PINN

↓

Id-Vg Prediction
```

을 수행한다.

이 논문는

```
Device Physics

↓

Physics-guided AI

↓

Compact Model
```

을 수행한다.

즉,

현재 프로젝트보다

한 단계 위의

Device Modeling 수준을 다루고 있다.

---

### Relation to My Career

공정 변화

↓

물성 변화

↓

Gamma 변화

↓

AI 보상

이라는 접근 방식은

이 논문의

Physics-guided AI Framework와 매우 유사하다고 느꼈다.

차이점은

Display Physics냐

Ferroelectric Physics냐의 차이뿐이다.

---

### Relation to SK hynix TCAD

```
Physics

↓

AI

↓

Compact Model

↓

Circuit

```

이라는 전체 Pipeline은

향후 AI 기반 반도체 연구개발의 핵심 방향이 될 가능성이 높다.

---

### Comparison with Previous Papers

| Paper | AI 역할 | Physics 활용 |
|---------|-----------------------------|----------------|
| S-ANN | TCAD Emulator | 없음 |
| PINN TCAD | TCAD Solver 보조 | 매우 높음 |
| NeuroSPICE | Circuit Solver | 매우 높음 |
| Physics-guided Compact Model | Compact Model 자동 생성 | 매우 높음 |

이 논문는

기존 리뷰한 논문들의

중간 역할을 수행한다.

```
TCAD

↓

Physics-guided AI

↓

Compact Model

↓

SPICE

↓

Circuit
```

즉,

Device Physics와 Circuit Modeling을 연결하는 핵심 기술이다.

---

### Future Learning Plan

이 논문를 기반으로 앞으로 다음 프로젝트를 진행할 계획이다.

- [ ] Compact Model 개념 학습 (BSIM, PSP, EKV)
- [ ] TFT GCA 기반 Compact Model 구현
- [ ] PINN 기반 Compact Model 생성 실험
- [ ] TCAD Dataset → Compact Model 자동 생성
- [ ] SPICE 연동 실험
- [ ] Device → Compact Model → Circuit Workflow 구현

---

### Paper Link

https://ieeexplore.ieee.org/document/10919462
