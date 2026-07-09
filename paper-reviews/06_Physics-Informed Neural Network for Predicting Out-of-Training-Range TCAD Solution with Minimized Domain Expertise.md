# Physics-Informed Neural Network for Predicting Out-of-Training-Range TCAD Solution with Minimized Domain Expertise

> **Takahiro Kubo et al. (2024)**
>
> **Physics-Informed Neural Network for Predicting Out-of-Training-Range TCAD Solution with Minimized Domain Expertise**
>
> **arXiv Preprint**

---

## Why I Read This Paper

반도체 TCAD 데이터를 이용해 AI 모델을 학습하면 일반적으로 **학습 데이터 범위를 벗어난 영역(Out-of-Training Range)** 에서는 예측 성능이 급격히 저하됩니다.

예를 들어,

- 0.2V ~ 0.8V 영역만 학습한 모델이
- 1.2V 영역을 예측해야 하는 경우

대부분의 Deep Learning 모델은 물리 법칙을 모르기 때문에 잘못된 결과를 출력합니다.

이 논문은 **PINN(Physics-Informed Neural Network)** 을 이용하여 최소한의 Domain Knowledge만으로도 학습 범위를 벗어난 TCAD 결과를 안정적으로 예측하는 방법을 제안합니다.

특히 실제 TCAD 결과를 대상으로 PINN의 일반화 성능을 검증했다는 점에서 매우 흥미로운 논문이었습니다.

---

## Problem

기존 Machine Learning 기반 TCAD Surrogate는 대부분

```
Training Data

↓

Deep Learning

↓

Interpolation
```

에는 강하지만

```
Training Range 밖

↓

Extrapolation
```

에는 매우 약합니다.

그 이유는

AI가

- Poisson Equation
- Drift-Diffusion Equation
- Carrier Physics

등을 이해하지 못하기 때문입니다.

따라서

새로운 Bias

새로운 Geometry

새로운 Process Condition

에서는 성능이 크게 저하됩니다.

기존 문제를 해결하기 위해서는

많은 Domain Knowledge

또는

매우 많은 TCAD Data

가 필요했습니다.

---

## Key Idea

이 논문의 핵심은

**Physics를 Loss Function에 포함하여**

학습 데이터가 없는 영역에서도

물리적으로 타당한 해를 찾도록 만드는 것입니다.

즉

기존 방식은

```
Data

↓

Neural Network

↓

Prediction
```

이라면

PINN은

```
Data

+

Physics Constraint

↓

Neural Network

↓

Physics-consistent Prediction
```

을 수행합니다.

이 논문에서는

Physics Constraint를 최대한 단순하게 설계하여

전문적인 Device Physics 지식이 없어도

PINN을 쉽게 적용할 수 있도록 한 것이 가장 큰 특징입니다.

---

## Method

전체 과정은 다음과 같습니다.

### Step 1. TCAD Dataset 생성

Sentaurus TCAD를 이용하여

MOSFET Device의

- Potential
- Carrier Density
- Electric Field

등을 계산합니다.

학습 데이터는 일부 Bias 조건에서만 생성합니다.

---

### Step 2. PINN 구성

입력

- Position (x, y)
- Bias Voltage

출력

- Potential
- Carrier Density

Neural Network를 구성합니다.

---

### Step 3. Physics Loss 추가

Loss는

```
Total Loss

=

Data Loss

+

Physics Loss
```

로 구성됩니다.

Physics Loss는

TCAD를 지배하는 PDE Residual을 최소화하도록 설계됩니다.

즉

Physics Equation을 만족하는 방향으로 Network를 학습합니다.

---

### Step 4. Out-of-Training Prediction

학습하지 않은 Bias 영역에서도

Physics Constraint 덕분에

물리적으로 일관된 결과를 예측합니다.

---

## Equation

PINN의 Loss는

\[
L
=
L_{data}
+
\lambda L_{physics}
\]

로 정의됩니다.

여기서

Data Loss

\[
L_{data}
=
MSE
(y_{pred},y_{true})
\]

Physics Loss는

Poisson Equation Residual을 이용합니다.

\[
R(x)
=
\nabla \cdot
(\epsilon \nabla V)
+\rho
\]

PINN은

\[
L_{physics}
=
MSE(R(x),0)
\]

을 최소화하도록 학습됩니다.

즉

예측 결과가

물리 방정식을 만족하도록

Loss를 구성합니다.

---

## Advantages

- 학습하지 않은 Bias 영역에서도 높은 정확도
- 기존 DNN보다 우수한 일반화 성능
- 적은 학습 데이터로도 안정적인 예측
- Physics를 이용하여 비현실적인 예측 감소
- TCAD Data 생성 비용 절감 가능
- Domain Knowledge 의존도를 최소화

---

## Limitations

- Physics Loss 계산 비용 증가
- PDE Residual 계산이 필요
- 매우 복잡한 Device에서는 Physics 식 정의가 어려울 수 있음
- Hyperparameter(λ)에 민감
- 완전한 TCAD Solver를 대체하기에는 아직 한계 존재

---

## Key Takeaways

- PINN은 단순 Curve Fitting이 아니라 Physics를 학습하는 모델이다.
- Physics Constraint는 학습 범위를 넘어서는 일반화 성능을 크게 향상시킨다.
- TCAD Dataset을 모두 생성하지 않아도 AI 모델 구축이 가능하다.
- 적은 Domain Expertise만으로도 PINN을 적용할 수 있도록 단순화한 것이 이 논문의 가장 큰 기여이다.
- Virtual R&D에서는 Simulation Cost 절감뿐 아니라 데이터 생성 비용 절감도 매우 중요하다.

---

## Personal Notes

### Relation to My Current Project

현재 GitHub에서 진행 중인 **GCA 기반 TFT PINN 프로젝트** 역시 같은 철학을 가지고 있습니다.

현재 프로젝트에서는

```
GCA Equation

↓

Physics Loss

↓

PINN

↓

Id-Vg Prediction
```

을 구현하고 있습니다.

이 논문는

```
TCAD PDE

↓

Physics Loss

↓

PINN

↓

Potential Prediction
```

이라는 차이만 있을 뿐

전체 구조는 거의 동일합니다.

현재 프로젝트가

PINN의 기초 구현이라면,

이 논문는 실제 반도체 Device에 적용한 산업 사례라고 볼 수 있습니다.

---

### Relation to My Career

제가 수행했던 업무 역시

```
소자 물성 분석

↓

비선형 모델링

↓

AI 모델

↓

양산 적용
```

이라는 흐름이었습니다.

이 논문에서는

```
TCAD Physics

↓

PINN

↓

Out-of-Range Prediction

↓

Virtual R&D
```

로 확장됩니다.

즉,

제가 기존에 수행했던

"물리 기반 모델링"

경험이

Scientific Machine Learning으로 자연스럽게 이어질 수 있다는 점을 확인할 수 있었습니다.

---

### Relation to SK hynix TCAD

SK하이닉스 TCAD에서는

- Physics AI
- Scientific Machine Learning
- Virtual R&D

를 강조하고 있습니다.

이 논문는

그 세 가지를 가장 잘 보여주는 사례 중 하나입니다.

특히

**적은 TCAD 데이터만으로도 새로운 공정 조건을 예측**

할 수 있다는 점은

향후

- 공정 최적화
- Device Optimization
- Parameter Exploration

등에서 매우 큰 가치가 있을 것으로 생각합니다.

---

### Future Learning Plan

이 논문를 기반으로 앞으로 다음 프로젝트를 진행할 계획입니다.

- [ ] 현재 TFT PINN 프로젝트에 Out-of-Training Validation 추가
- [ ] Bias Extrapolation 성능 비교
- [ ] Pure DNN vs PINN 비교 실험
- [ ] 다양한 Physics Loss Weight(λ) 분석
- [ ] TCAD Dataset 기반 PINN 구현
- [ ] Device Parameter Sweep 자동화

---

### Paper Link

https://arxiv.org/abs/2408.07921