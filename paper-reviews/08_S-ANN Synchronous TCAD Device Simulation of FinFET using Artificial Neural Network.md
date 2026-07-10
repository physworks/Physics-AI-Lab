# S-ANN: Synchronous TCAD Device Simulation of FinFET using Artificial Neural Network

> Yansen Liu et al. (2025)
>
> S-ANN: Synchronous TCAD Device Simulation of FinFET using Artificial Neural Network
>
> Microelectronics Journal

---

## Why I Read This Paper

최근 반도체 산업에서는 AI를 이용하여 TCAD 시뮬레이션을 대체하거나 가속하는 연구가 활발하게 진행되고 있다.

이 논문은 Physics-Informed AI가 아닌 **순수 Artificial Neural Network(ANN)** 를 이용하여 FinFET TCAD를 대체하는 Surrogate Model을 구축한 연구이다.

특히 기존 Sentaurus TCAD의 반복 계산을 수행하지 않고도 Device 특성을 빠르게 예측할 수 있는 AI 기반 TCAD Emulator를 제안했다는 점에서 현재 진행 중인 Physics-AI-Lab 프로젝트와 매우 밀접한 관련이 있다고 판단하여 읽게 되었다.

---

## Problem

기존 TCAD(Device Simulation)는 매우 높은 정확도를 제공하지만 계산 시간이 오래 걸린다.

FinFET과 같은 3차원 소자의 경우에는

- Device Mesh 생성
- Poisson Equation
- Drift-Diffusion Equation
- Carrier Transport
- Newton Iteration

등을 반복적으로 계산해야 한다.

Parameter Sweep만 수행하더라도 수천 번의 Simulation이 필요하며,

공정 최적화(Process Optimization)

DTCO(Design Technology Co-Optimization)

Technology Exploration

등에서는 계산 시간이 가장 큰 병목이 된다.

또한 TCAD는 Solver Convergence Failure가 발생할 수 있으며,

많은 계산 자원이 요구된다.

---

## Key Idea

논문의 핵심 아이디어는

**TCAD Solver를 직접 수행하지 않고 AI가 Device Physics 결과를 학습하도록 만드는 것이다.**

기존 Workflow

```
Device Structure

↓

Sentaurus TCAD

↓

Potential

Carrier Density

IV Characteristics

↓

Device Analysis
```

제안한 Workflow

```
Device Parameters

↓

Artificial Neural Network

↓

Potential

Carrier Density

IV Characteristics

↓

Device Analysis
```

즉,

TCAD Solver 자체를 하나의 Neural Network로 근사하는

**TCAD Surrogate Model**을 구축하였다.

---

## Method

### Step 1. TCAD Dataset 생성

Sentaurus TCAD를 이용하여

다양한

- Gate Length
- Fin Width
- Fin Height
- Bias Voltage

조건에 대해 Simulation을 수행하였다.

생성된 Dataset에는

- Potential
- Electron Density
- Hole Density
- Electric Field
- Drain Current

등의 Device Physics 정보가 포함된다.

---

### Step 2. Dataset 구성

입력(Input)

- Device Geometry
- Material Parameter
- Bias Condition

출력(Output)

- Potential Distribution
- Carrier Distribution
- IV Characteristics

모든 데이터는 Normalization을 수행하여 학습 안정성을 높였다.

---

### Step 3. ANN 구축

TensorFlow 기반 Fully Connected ANN을 사용하였다.

구조는 비교적 단순하다.

- Hidden Layer 3개
- 각 Layer당 10개의 Neuron

복잡한 모델보다

품질 좋은 TCAD Dataset 구축을 더 중요하게 보았다.

---

### Step 4. TCAD Emulator

학습이 완료되면

새로운 Device Parameter가 입력되었을 때

ANN이 즉시

- Potential
- Carrier Density
- IV Curve

를 예측한다.

따라서

기존 TCAD Solver를 반복 수행할 필요가 없다.

---

## Equation

ANN은 다음과 같은 함수를 학습한다.

\[
f_\theta(x)=y
\]

입력

\[
x=(Geometry,\ Material,\ Bias)
\]

출력

\[
y=(Potential,\ Carrier,\ Current)
\]

Loss Function은 일반적인 Mean Squared Error(MSE)를 사용하였다.

\[
L=\frac{1}{N}\sum_i(y_i-\hat y_i)^2
\]

Physics Equation은 Loss에 포함되지 않는다.

즉,

완전한 Data-driven Learning이다.

---

## Advantages

- TCAD Simulation보다 매우 빠른 추론 속도
- 반복적인 Parameter Sweep에 적합
- TCAD Solver의 Convergence 문제 제거
- Device Optimization 시간 단축
- 기존 Sentaurus Workflow와 쉽게 연동 가능
- 구현이 단순하여 산업 적용이 용이함

---

## Limitations

- 많은 TCAD Dataset 생성이 필요하다.
- Physics Constraint가 존재하지 않는다.
- Training Range 밖에서는 성능 저하 가능성이 있다.
- 새로운 Device 구조에서는 재학습이 필요하다.
- PINN보다 물리적 해석성이 낮다.

---

## Key Takeaways

- TCAD를 AI로 근사하는 가장 기본적인 접근법이다.
- 좋은 AI보다 좋은 Dataset이 더욱 중요하다.
- Physics를 사용하지 않아도 실제 산업에서는 충분히 활용 가능하다.
- Surrogate Model은 Parameter Sweep을 크게 가속할 수 있다.
- AI 기반 Virtual R&D의 첫 단계는 이러한 TCAD Emulator 구축이다.

---

## Personal Notes

### Relation to My Current Project

진행했었던 **GCA 기반 TFT PINN 프로젝트**는

```
Physics Equation

↓

PINN

↓

Id-Vg Prediction
```

구조를 사용한다.

반면

S-ANN은

```
TCAD Dataset

↓

ANN

↓

Device Prediction
```

을 수행한다.

현재 프로젝트보다 Physics 정보는 적지만,

산업에서 실제 구축하기 쉬운 구조라는 점이 인상적이었다.

---

### Relation to My Career

S-ANN 역시

```
TCAD Simulation

↓

Dataset 구축

↓

AI 모델

↓

Virtual Device
```

라는 동일한 개발 철학을 가지고 있다.

Physics를 이해하고,

데이터를 구축하며,

AI를 적용한다는 점에서

기존 업무 경험과 매우 유사하다고 느꼈다.

---

### Relation to SK hynix TCAD

SK하이닉스 TCAD 조직은

- Material Simulation
- Scientific Machine Learning
- Virtual R&D

를 핵심 방향으로 제시하고 있다.

S-ANN은

그중에서도

**Simulation Acceleration**

을 직접 수행하는 대표적인 사례이다.

다만 향후에는

Physics Constraint가 포함된

PINN이나 Neural Operator 기반 모델이

더 높은 일반화 성능을 제공할 것으로 예상된다.

---

### Future Learning Plan

이 논문를 기반으로 앞으로 다음 프로젝트를 진행할 계획이다.

- [ ] Sentaurus 결과 기반 TCAD Dataset 구축
- [ ] ANN 기반 TCAD Surrogate Model 구현
- [ ] PINN과 ANN 성능 비교
- [ ] Bias Sweep 자동화
- [ ] Parameter Sweep 자동화
- [ ] Neural Operator와 성능 비교

---

### Paper Link

https://doi.org/10.1016/j.mejo.2025.106630