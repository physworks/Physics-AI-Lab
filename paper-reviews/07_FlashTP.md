# FlashTP

> **Lee et al. (2025)**
>
> **FlashTP: Fused, Sparsity-Aware Tensor Product for Machine Learning Interatomic Potentials**
>
> **International Conference on Machine Learning (ICML 2025)**

---

## Why I Read This Paper

SK하이닉스 TCAD 직무에서는 **Material Simulation**, **Machine Learning Interatomic Potential (MLIP)**, **Physics AI** 역량이 중요한 키워드이다.

최근 MLIP는 DFT(밀도범함수이론)와 Molecular Dynamics(MD)를 대체하거나 가속하는 핵심 기술로 자리 잡고 있으며, 대표적인 모델인 **NequIP**, **Allegro**, **MACE**, **SevenNet** 등은 모두 **Equivariant Graph Neural Network**를 기반으로 한다.

하지만 이러한 모델들의 가장 큰 병목은 모델 구조가 아니라 **Tensor Product 연산**이다.

FlashTP는 새로운 MLIP 모델을 제안하는 논문이 아니라, **기존 Equivariant MLIP를 GPU에서 훨씬 빠르게 실행하기 위한 시스템 최적화 기술**을 제안한다.

즉, Physics AI와 GPU 시스템 최적화가 결합된 대표적인 연구이다.

---

## Problem

Equivariant MLIP는 회전 대칭성(SO(3) Equivariance)을 유지하기 위해 Tensor Product 연산을 반복적으로 수행한다.

대표적인 모델은 다음과 같다.

- NequIP
- Allegro
- SevenNet
- MACE
- e3nn 기반 모델

이들 모델에서는 Tensor Product 연산이

- Clebsch-Gordan(CG) coefficient 계산
- 많은 수의 작은 CUDA Kernel 실행
- Intermediate Tensor 생성
- Sparse coefficient 처리

등으로 인해 전체 학습 시간의 대부분을 차지한다.

논문에서는 Tensor Product 연산이 전체 실행 시간의 약 **75%**를 소비하며, GPU Memory Traffic 또한 매우 크게 증가한다고 분석한다.

결국 좋은 Physics Model을 가지고 있어도 GPU 활용 효율이 매우 낮다는 것이 핵심 문제이다.

---

## Key Idea

FlashTP의 핵심은 Tensor Product의 **수학적 정의는 그대로 유지하면서 GPU 실행 방식을 완전히 다시 설계하는 것**이다.

핵심 아이디어는 세 가지이다.

### 1. Kernel Fusion

기존 방식

```
Tensor Product
    ↓
Activation
    ↓
Linear
    ↓
Reduction
```

각 단계마다 별도의 CUDA Kernel을 실행하였다.

FlashTP는 이를

```
Single Fused CUDA Kernel
```

하나로 통합하여 중간 Tensor 생성을 제거한다.

이를 통해 Memory Access와 Kernel Launch Overhead를 크게 줄인다.

---

### 2. Sparse Tensor Product

Tensor Product에 사용되는 Clebsch-Gordan coefficient는 대부분이 0이다.

기존 구현은 모든 coefficient를 계산하지만,

FlashTP는 **0이 아닌 coefficient만 계산**하도록 Sparse 구조를 활용한다.

즉,

```
Dense Computation

↓

Sparse Computation
```

으로 변경하여 계산량을 크게 줄인다.

---

### 3. Path Aggregation

Tensor Product는 여러 개의 Coupling Path를 독립적으로 계산한다.

FlashTP는 유사한 Path들을 하나의 CUDA Kernel 안에서 동시에 계산하여

- GPU Occupancy 증가
- Kernel Launch 감소
- Memory Access 감소

를 달성한다.

---

## Method

FlashTP의 전체 연산 흐름은 다음과 같다.

기존 방식

```
Input

↓

Neighbor Message

↓

Tensor Product

↓

Linear

↓

Output
```

실제 GPU 실행은

```
Kernel A

↓

Memory Write

↓

Kernel B

↓

Memory Write

↓

Kernel C
```

형태였다.

FlashTP에서는

```
Input

↓

Single Fused Kernel

↓

Output
```

으로 변경된다.

또한 Clebsch-Gordan Matrix의 Sparse Pattern을 Compile Time에 분석하여 Runtime에서 불필요한 Branch와 Memory Access를 최소화하였다.

---

## Equation

Equivariant MLIP에서 핵심 Tensor Product는

\[
z = x \otimes y
\]

으로 표현된다.

실제 구현은 Clebsch-Gordan coefficient를 이용하여

\[
z_i
=
\sum_{j,k}
C_{ijk}
x_j
y_k
\]

형태로 계산된다.

여기서

- \(C_{ijk}\) : Clebsch-Gordan coefficient
- 대부분의 \(C_{ijk}\)는 0

FlashTP는 수식을 변경하지 않고,

**0이 아닌 coefficient만 계산하는 Sparse Tensor Product**를 구현하였다.

즉,

Physics는 그대로 유지하면서 GPU 계산량만 크게 감소시킨 것이 핵심이다.

---

## Advantages

### 1. 매우 큰 GPU Kernel 가속

Tensor Product Kernel 기준

- 최대 **41.6×** (e3nn 대비)
- 최대 **60.8×** (NVIDIA cuEquivariance 대비)

속도 향상을 달성하였다.

---

### 2. 실제 MLIP 학습 속도 향상

SevenNet 기준

- Inference : **4.2×**
- Training : **3.5×**

속도 향상을 보였다.

---

### 3. Memory 사용량 감소

Peak GPU Memory를 약 **6배 이상 감소**시켜 대형 MLIP 모델 학습이 가능해졌다.

---

### 4. 기존 모델 그대로 사용 가능

FlashTP는 새로운 ML 모델이 아니다.

기존

- NequIP
- Allegro
- MACE
- SevenNet

등 대부분의 Equivariant MLIP에 쉽게 적용 가능하다.

---

### 5. 정확도 변화 없음

Tensor Product의 수학적 정의는 그대로 유지한다.

계산 방식만 최적화하기 때문에 Accuracy 손실이 없다.

---

## Limitations

### 1. 새로운 Physics Model은 아니다.

Physics를 개선하는 논문가 아니라 GPU 시스템 최적화 논문이다.

---

### 2. CUDA 의존성이 높다.

GPU Kernel 최적화가 핵심이므로 CPU에서는 장점이 거의 없다.

---

### 3. Tensor Product 기반 모델에서만 사용 가능하다.

Transformer나 일반 CNN, PINN에는 직접 적용하기 어렵다.

Equivariant MLIP 전용 기술이다.

---

### 4. Hardware Optimization 성격이 강하다.

새로운 AI 알고리즘보다는 HPC(System Optimization)에 가까운 연구이다.

---

## Key Takeaways

FlashTP의 가장 중요한 메시지는

> **MLIP가 느린 이유는 Physics 때문이 아니라 Tensor Product 구현 때문이라는 것이다.**

Physics를 바꾸지 않아도

GPU Kernel만 최적화하여

- 최대 **60배 Kernel Speedup**
- 약 **4배 추론 속도 향상**
- 약 **6배 Memory 절감**

을 달성하였다.

최근 Physics AI에서는

**좋은 모델을 만드는 것만큼 실제 실행 속도를 높이는 것이 매우 중요**하다는 흐름을 보여주는 대표적인 논문이다.

---

## Personal Notes

FlashTP는 새로운 Physics Model을 만드는 것이 아니라,

**동일한 계산 결과를 훨씬 빠르게 수행하도록 시스템을 최적화**한다.

즉,

```
Display Gamma Optimization

↓

Physics AI Optimization

↓

MLIP Runtime Optimization
```

이라는 커리어 연결성을 만들 수 있다.

또한 향후 TCAD 및 Physics AI 분야에서는

- PINN Acceleration
- Neural Operator Inference
- MLIP Runtime Optimization
- GPU Kernel Optimization
- Physics Simulator Acceleration

등이 모두 같은 방향으로 발전하고 있다.

FlashTP는 Physics AI 모델 자체를 이해하기 위한 논문이라기보다,

**Physics AI를 실제 산업에서 사용할 수 있는 수준으로 가속하는 기술을 이해하기 위한 핵심 논문**이라고 생각한다.


+ 학부 때, 많은 가르침을 받고 졸업 논문까지 도움받은 이재욱 교수님 연구실의 논문이라 더 몰입되고 재밌게 리뷰하였다.
시간될 떄 한번 찾아뵙고 인사드려야겠다.

---

## Paper Link

- https://proceedings.mlr.press/v267/lee25l.html
- https://openreview.net/forum?id=wiQe95BPaB