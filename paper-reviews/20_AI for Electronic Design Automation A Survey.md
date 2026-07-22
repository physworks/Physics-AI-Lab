# AI for Electronic Design Automation: A Survey

> Huang et al. (2021)
>
> Machine Learning for Electronic Design Automation: A Survey
>
> ACM Transactions on Design Automation of Electronic Systems (TODAES)

---

## Why I Read This Paper

반도체 미세화가 진행될수록 설계 공간(Design Space)은 기하급수적으로 증가하고 있으며, 기존 Rule-based EDA만으로는 최적 설계를 수행하기 어려워지고 있다.

이 논문은 Machine Learning이 EDA 전 과정에서 어떻게 활용되는지를 체계적으로 정리한 대표적인 Survey 논문이다.

비록 SK하이닉스 TCAD 직무는 설계(EDA)보다 Device Simulation에 가깝지만, AI가 반도체 설계와 시뮬레이션 전반을 어떻게 변화시키고 있는지 이해하기 위해 읽었다. :contentReference[oaicite:0]{index=0}

---

## Problem

기존 EDA Flow는 다음과 같다.

```
Specification

↓

RTL Design

↓

Logic Synthesis

↓

Placement

↓

Routing

↓

Timing Analysis

↓

Verification
```

각 단계마다

- NP-Hard Optimization
- 반복적인 탐색(Search)
- 긴 Runtime
- 많은 Expert Rule

이 필요하다.

공정이 미세화될수록

- Timing Closure
- Power Optimization
- Congestion
- DRC

등이 매우 복잡해지며 설계 시간이 급격히 증가한다. :contentReference[oaicite:1]{index=1}

---

## Key Idea

EDA의 각 단계를

Rule-based Optimization 대신

Machine Learning Problem으로 바꾼다.

즉,

```
EDA Optimization

↓

Machine Learning

↓

Prediction

↓

Optimization
```

으로 전환한다.

AI는

- Layout 예측
- Congestion 예측
- Delay 예측
- Power 예측
- DRC 예측
- Routing 최적화

등을 수행하여 설계 반복 횟수를 줄인다. :contentReference[oaicite:2]{index=2}

---

## Method

### Step 1. Feature Extraction

EDA 단계에서

- Cell
- Net
- Timing
- Layout
- Graph

등의 정보를 Feature로 추출한다.

---

### Step 2. ML Model

문제에 따라

- Random Forest
- XGBoost
- CNN
- GNN
- Reinforcement Learning
- Bayesian Optimization

등 다양한 모델을 적용한다.

---

### Step 3. Prediction

AI는

- Delay
- Power
- Routing Cost
- Congestion
- Yield

등을 예측한다.

---

### Step 4. Optimization

예측 결과를 이용하여

EDA Tool이

보다 좋은 Solution을 빠르게 탐색한다.

즉

AI가 설계를 직접 수행하는 것이 아니라

EDA Solver를 Guide하는 역할을 수행한다. :contentReference[oaicite:3]{index=3}

---

## Equation

EDA Optimization은 일반적으로

```
min f(x)
```

형태의 최적화 문제이다.

여기서

- x : Design Variable
- f : Area / Power / Delay / Cost

AI는

\[
\hat{f}(x)
\]

를 근사하여

비용이 큰 실제 Evaluation 대신

빠른 Prediction을 수행한다.

또한 Reinforcement Learning에서는

\[
\pi(a|s)
\]

를 학습하여

최적의 Design Action을 선택한다. :contentReference[oaicite:4]{index=4}

---

## Advantages

- EDA Runtime 단축
- Design Space Exploration 가속
- Timing Closure 향상
- Congestion 감소
- Routing 효율 향상
- 반복 설계 비용 감소

---

## Limitations

- 대규모 학습 데이터가 필요하다.
- 새로운 공정(Node)에서는 재학습이 필요하다.
- 설계 규칙 변경 시 일반화가 어렵다.
- 산업 데이터가 공개되지 않아 연구 재현성이 낮다.
- 대부분의 모델은 특정 EDA 단계에 특화되어 있다. :contentReference[oaicite:5]{index=5}

---

## Key Takeaways

- AI는 EDA Tool을 대체하는 것이 아니라 Optimization Engine을 보조한다.
- EDA의 거의 모든 단계에 Machine Learning이 적용되고 있다.
- GNN, RL, Bayesian Optimization이 특히 많이 활용된다.
- 반도체 설계 자동화는 점점 AI 중심으로 발전하고 있다.
- AI 기반 EDA는 Virtual R&D의 핵심 요소 중 하나이다. :contentReference[oaicite:6]{index=6}

---

## Personal Notes

### Relation to My Current Project

현재 진행 중인 프로젝트는

```
Physics

↓

PINN

↓

Neural Operator

↓

TCAD Surrogate
```

이다.

반면 이 논문은

```
Circuit Design

↓

Machine Learning

↓

EDA Optimization
```

을 다룬다.

즉,

AI 적용 대상이

Physics Simulation이 아니라

Design Optimization이라는 차이가 있다.

---

### Relation to SK hynix TCAD

직접적인 TCAD 논문는 아니지만

AI를 활용하여

복잡한 Optimization 문제를 해결한다는 점은

TCAD Virtual R&D와 매우 유사하다.

특히

- Surrogate Model
- Fast Prediction
- Optimization Loop

라는 개념은

TCAD에도 그대로 적용된다.

---

### Comparison with Previous Papers

| Paper | Target |
|---------|-------------------------|
| PINN | PDE Solver |
| DeepONet | Operator Learning |
| FNO | PDE Surrogate |
| MeshGraphNet | Mesh Simulation |
| MLIP | Atomic Potential |
| TCAD Surrogate | Device Simulation |
| **ML for EDA Survey** | Circuit Design Optimization |

---

### Why This Paper Matters

이 논문는 Physics AI를 다루지는 않지만,

반도체 산업 전체가

```
Rule-based Engineering

↓

Machine Learning

↓

AI-assisted Engineering

↓

Foundation Models
```

방향으로 발전하고 있음을 잘 보여준다.

최근에는 여기에 LLM과 Agent 기반 EDA까지 더해지는 추세다. :contentReference[oaicite:7]{index=7}

---

### Future Learning Plan

- [ ] Reinforcement Learning for Placement
- [ ] Graph Neural Network for Routing
- [ ] Bayesian Optimization
- [ ] AI Placement & Routing
- [ ] AI Timing Prediction
- [ ] AI 기반 Design Space Exploration 학습

---

### Paper Link

https://arxiv.org/abs/2102.03357