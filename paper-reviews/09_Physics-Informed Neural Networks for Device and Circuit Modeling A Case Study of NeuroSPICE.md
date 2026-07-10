# Physics-Informed Neural Networks for Device and Circuit Modeling: A Case Study of NeuroSPICE

> Chien-Ting Tung, Chenming Hu (2025)
>
> **Physics-Informed Neural Networks for Device and Circuit Modeling: A Case Study of NeuroSPICE**
>
> IEEE Electron Device Letters (Preprint)

---

## Why I Read This Paper

기존 SPICE는 반도체 회로 설계에서 가장 널리 사용되는 회로 시뮬레이터이지만, 새로운 소자(FeRAM, RRAM, FTJ, Photonics 등)가 등장할수록 새로운 Compact Model을 개발하고 Verilog-A로 구현하는 과정이 점점 복잡해지고 있다.

이 논문은 **Physics-Informed Neural Network(PINN)** 를 이용하여 기존 SPICE Solver 자체를 Neural Network로 대체하는 **NeuroSPICE**를 제안한다.

특히 Device Physics뿐만 아니라 Circuit Simulation까지 PINN으로 해결하려는 첫 사례 중 하나로, 향후 AI 기반 EDA 및 Virtual R&D가 어떤 방향으로 발전할 것인지 이해하기 위해 읽게 되었다. :contentReference[oaicite:0]{index=0}

---

## Problem

기존 SPICE는 Modified Nodal Analysis(MNA)를 기반으로 다음 과정을 반복 수행한다.

```
Circuit Netlist

↓

Differential Algebraic Equation (DAE)

↓

Time Discretization

↓

Newton-Raphson Iteration

↓

Waveform
```

이 방식은 매우 정확하지만 다음과 같은 한계가 존재한다.

- 시간(Time Step)을 매우 작게 설정해야 한다.
- 강한 비선형 소자에서는 수렴(Convergence)이 어려울 수 있다.
- 새로운 소자 모델을 Verilog-A로 구현해야 한다.
- Multiphysics 모델을 추가하기 어렵다.
- Gradient 기반 최적화와 직접 연결하기 어렵다.

특히 Ferroelectric Memory와 같은 Emerging Device에서는 이러한 문제가 더욱 심해진다. :contentReference[oaicite:1]{index=1}

---

## Key Idea

논문의 핵심 아이디어는

**회로 미분방정식(DAE)을 PINN의 Loss Function으로 직접 사용하는 것**이다.

기존 SPICE는

```
Circuit

↓

Numerical Solver

↓

Waveform
```

이지만,

NeuroSPICE는

```
Circuit Equation

↓

Physics-Informed Neural Network

↓

Continuous Waveform
```

을 수행한다.

즉,

Neural Network가

- Node Voltage
- Branch Current

를 시간의 연속 함수(Continuous Function)로 직접 학습한다.

따라서 별도의 Time Stepping 없이도 회로 파형을 계산할 수 있다. :contentReference[oaicite:2]{index=2}

---

## Method

### Step 1. Circuit Equation 정의

회로를

Differential Algebraic Equation(DAE)

형태로 표현한다.

예를 들어

```
Capacitor

Current Source

Transistor

Resistor
```

등의 관계식을 하나의 시스템 방정식으로 구성한다.

---

### Step 2. PINN 구성

입력

- Time

출력

- Node Voltage
- Branch Current

Neural Network는

시간을 입력받아

회로 전체 상태를 출력한다.

---

### Step 3. Physics Loss 계산

PINN은

회로 방정식 Residual을 Loss Function으로 사용한다.

Loss는

```
Total Loss

=

Initial Condition

+

Boundary Condition

+

Circuit Residual
```

로 구성된다.

Backpropagation을 이용하여

시간에 대한 미분을 정확하게 계산한다.

Finite Difference를 사용할 필요가 없다. :contentReference[oaicite:3]{index=3}

---

### Step 4. Circuit Simulation

학습이 완료되면

Neural Network는

연속적인 시간 함수 형태로

- Voltage
- Current

를 즉시 계산한다.

논문에서는

- Transistor Amplifier
- Ring Oscillator
- Ferroelectric Memory

를 대상으로 검증하였다. :contentReference[oaicite:4]{index=4}

---

## Equation

NeuroSPICE는

Circuit DAE를 만족하도록 학습한다.

일반적인 형태는

\[
F(x,\dot{x},t)=0
\]

이다.

PINN은

\[
x(t)
=
NN_\theta(t)
\]

를 학습한다.

Loss는

\[
L
=
L_{IC}
+
L_{BC}
+
L_{Physics}
\]

이며,

Physics Loss는

\[
L_{Physics}
=
\sum
\left|
F(x,\dot{x},t)
\right|^2
\]

이다.

즉,

회로 미분방정식을 만족하는 방향으로

Neural Network를 학습한다. :contentReference[oaicite:5]{index=5}

---

## Advantages

- Time Discretization이 필요 없다.
- Time Step Error가 발생하지 않는다.
- Time Derivative를 Auto-differentiation으로 정확하게 계산한다.
- Verilog-A 없이 새로운 Device를 빠르게 적용할 수 있다.
- Multiphysics 시스템 확장이 쉽다.
- Gradient 기반 Design Optimization과 자연스럽게 연결된다.
- Inverse Problem 해결에 적합하다.

---

## Limitations

- 학습 시간이 기존 SPICE보다 길다.
- 정확도 역시 기존 SPICE를 완전히 대체하지는 못한다.
- Large Circuit으로 확장 시 Scalability 검증이 필요하다.
- PINN 학습 안정성이 중요하다.
- Loss Weight 조정이 어렵다.

논문에서도 **NeuroSPICE가 SPICE보다 빠르거나 정확한 것이 목적은 아니며**, Differentiable Simulator라는 새로운 장점에 의미가 있다고 설명한다. :contentReference[oaicite:6]{index=6}

---

## Key Takeaways

- PINN은 Device Simulation뿐 아니라 Circuit Simulation에도 적용 가능하다.
- 기존 SPICE Solver를 Neural Network로 대체할 수 있다.
- Time Stepping 없이 연속적인 Waveform을 계산한다.
- Gradient를 직접 계산할 수 있으므로 Design Optimization에 매우 유리하다.
- AI 기반 EDA의 새로운 방향을 제시하는 연구이다.

---

## Personal Notes

### Relation to My Current Project

진행했었던 **GCA 기반 TFT PINN**은

```
Physics Equation

↓

PINN

↓

Id-Vg Prediction
```

구조이다.

NeuroSPICE는

```
Circuit Equation

↓

PINN

↓

Voltage / Current
```

를 수행한다.

즉,

현재 프로젝트를

Device Level에서

Circuit Level로 확장한 개념이라고 볼 수 있다.

---

### Relation to My Career

NeuroSPICE 역시

```
Device Physics

↓

Circuit Equation

↓

PINN

↓

Circuit Simulator
```

라는 철학을 가지고 있다.

특히

Physics를 이해하고

이를 실제 동작하는 Software로 구현한다는 점에서

기존 업무 경험과 매우 유사하다고 느꼈다.

---

### Relation to SK hynix TCAD

SK하이닉스 TCAD 조직은

- Material Simulation
- Scientific Machine Learning
- Physics AI
- Virtual R&D

를 핵심 방향으로 제시하고 있다.

NeuroSPICE는

Device Simulation을 넘어

Circuit Simulation까지 AI 기반으로 확장한 사례이다.

향후

TCAD

↓

Compact Model

↓

Circuit Simulation

↓

EDA

전체 Workflow가

AI 기반으로 연결될 가능성을 보여주는 매우 의미 있는 논문이라고 생각한다.

---

### Future Learning Plan

이 논문를 기반으로 앞으로 다음 프로젝트를 진행할 계획이다.

- [ ] PINN 기반 RC Circuit 구현
- [ ] MOSFET Compact Model PINN 구현
- [ ] GCA 기반 TFT를 Circuit Level까지 확장
- [ ] PINN 기반 Ring Oscillator 구현
- [ ] Device PINN과 Circuit PINN 비교
- [ ] TCAD → Compact Model → Circuit PINN 전체 Workflow 구현

---

### Paper Link

https://arxiv.org/abs/2512.23624