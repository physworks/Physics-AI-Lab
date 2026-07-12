# Neural Operators as Data-Driven Surrogate Models for Partial Differential Equations

> Zongyi Li, Nikola Kovachki, Burigede Liu, Kamyar Azizzadenesheli, Anima Anandkumar, Andrew Stuart, et al. (2020)
>
> **Neural Operators as Data-Driven Surrogate Models for Partial Differential Equations**
>
> arXiv (2020)

---

## Why I Read This Paper

기존 PINN은 PDE를 만족하도록 학습하지만, 새로운 경계조건(Boundary Condition)이나 초기조건(Initial Condition)이 바뀔 때마다 새로운 학습이 필요하다는 한계가 있습니다.

반면 실제 TCAD에서는 다양한 공정 변수(CD, Thickness, Doping, Temperature, Material Property)에 대해 반복적으로 PDE를 풀어야 합니다.

이 논문은 **"PDE 자체를 푸는 것이 아니라 PDE Solution Operator를 학습한다."** 는 새로운 접근을 제안하며, 이후 DeepONet, Fourier Neural Operator(FNO), MeshGraphNet 등 현대 Scientific Machine Learning 연구의 출발점이 되었습니다.

SK하이닉스 TCAD가 추진하는 AI 기반 Virtual R&D에서도 수천~수만 번의 TCAD Simulation을 대체할 Surrogate Model이 핵심이므로 반드시 이해해야 하는 논문이라고 판단했습니다.

---

## Problem

기존 PDE Solver는

- Finite Difference Method (FDM)
- Finite Element Method (FEM)
- Finite Volume Method (FVM)

등을 사용하여 수치적으로 PDE를 반복 계산합니다.

문제는

- 계산량이 매우 크고
- Mesh가 촘촘할수록 계산 시간이 급격히 증가하며
- Parameter Sweep가 거의 불가능할 정도로 느리다는 것입니다.

PINN 역시 새로운 PDE 조건마다 다시 학습해야 하므로 범용성이 제한됩니다.

즉,

```
Input Condition

↓

PDE Solver

↓

Solution
```

을 매번 반복해야 하는 것이 가장 큰 문제입니다.

---

## Key Idea

이 논문의 핵심은

**Solution Function이 아니라 Solution Operator를 학습한다.**

기존 Neural Network는

```
x

↓

y
```

라는 하나의 함수를 학습합니다.

반면 Operator Learning은

```
Function

↓

Function
```

을 학습합니다.

즉,

입력

```
Initial Condition

Boundary Condition

Material Property
```

전체를 하나의 함수로 입력받아

출력 역시

```
Entire Solution Field
```

를 예측합니다.

이를 통해 새로운 PDE 조건이 들어와도 추가 학습 없이 바로 예측이 가능합니다.

---

## Method

### Step 1. PDE Dataset 생성

다양한 PDE 조건에 대해

- Boundary Condition
- Initial Condition
- Material Parameter

를 변경하며 수많은 Numerical Solution을 생성합니다.

예)

```
Poisson Equation

Heat Equation

Burgers Equation

Navier-Stokes
```

---

### Step 2. Operator Learning

Neural Network는

```
Input Function

↓

Operator

↓

Output Function
```

을 직접 학습합니다.

즉

```
u(x)

↓

G

↓

y(x)
```

를 근사합니다.

여기서

G가 Operator입니다.

---

### Step 3. Generalization

새로운

- Geometry
- Mesh
- Boundary Condition

에서도

Operator 자체를 학습했기 때문에

바로 예측 가능합니다.

---

## Equation

일반적인 PDE는

\[
F(u)=0
\]

형태로 표현됩니다.

기존 PINN은

특정 해

\[
u(x)
\]

를 학습합니다.

반면 Neural Operator는

Operator

\[
G
\]

를 학습합니다.

즉,

\[
G : a(x) \rightarrow u(x)
\]

여기서

- \(a(x)\) : 입력 함수 (Boundary Condition, Material Property 등)
- \(u(x)\) : PDE Solution

입니다.

Loss는

\[
L
=
\|G(a)-u\|^2
\]

를 최소화하도록 학습합니다.

---

## Advantages

- 새로운 Boundary Condition에서도 재학습이 필요 없음
- Entire Solution Field를 한 번에 예측 가능
- 기존 Solver보다 수백~수천 배 빠른 추론 가능
- Mesh Resolution 변화에도 비교적 강건함
- 다양한 PDE에 적용 가능
- Surrogate Model 구축에 적합
- TCAD Virtual R&D와 매우 높은 연관성

---

## Limitations

- 초기 Dataset 생성 비용이 매우 크다.
- 충분히 다양한 PDE 데이터를 확보해야 한다.
- 매우 복잡한 Multi-Physics에서는 일반화가 어려울 수 있다.
- 학습 범위를 크게 벗어난 조건에서는 정확도가 저하될 수 있다.
- Operator 구조가 아직 발전 중이며 구현 난이도가 높다.

---

## Key Takeaways

- PINN은 PDE를 만족하는 Solution을 학습한다.
- Neural Operator는 PDE의 Solution Operator 자체를 학습한다.
- 새로운 Boundary Condition에서도 추가 학습 없이 예측 가능하다.
- TCAD에서는 수많은 Parameter Sweep을 매우 빠르게 수행할 수 있다.
- 이후 DeepONet, Fourier Neural Operator, MeshGraphNet 연구의 기반이 된 논문이다.

---

## Personal Notes

### Relation to My Current Project

현재 GitHub에서 진행 중인

**TFT GCA PINN 프로젝트**는

```
Physics Equation

↓

PINN

↓

Id-Vg Prediction
```

을 수행한다.

하지만 앞으로 TCAD에서는

```
Structure

↓

TCAD

↓

Entire Device Field
```

를 예측해야 한다.

이때 필요한 것이 바로

Neural Operator이다.

---

### Relation to My Career

Input Physics

↓

Output Physics

를 학습하는 과정이었다.

Neural Operator는 이를

PDE 수준까지 확장한 개념이라고 이해했다.

---

### Relation to SK hynix TCAD

SK하이닉스 JD에서 언급한

- Scientific Machine Learning
- Physics AI
- Virtual R&D

의 핵심 기술 중 하나이다.

향후

```
TCAD Dataset

↓

Neural Operator

↓

Instant Simulation
```

형태의 Workflow가

차세대 Device Simulation의 표준이 될 가능성이 높다.

---

### Comparison with Previous Papers

| Paper | 학습 대상 | Physics 활용 |
|---------|------------------------------|----------------|
| PINN | PDE Solution | 매우 높음 |
| DeepONet | Solution Operator | 높음 |
| Fourier Neural Operator | Fourier Domain Operator | 높음 |
| Neural Operator | 일반 Operator Learning | 높음 |

Neural Operator는

Operator Learning이라는 새로운 개념을 제시한 논문이며,

이후 DeepONet과 FNO는 이를 더욱 발전시킨 구조라고 이해할 수 있다.

---

### Future Learning Plan

이 논문를 기반으로 다음 프로젝트를 진행할 계획이다.

- [x] DeepONet 구현
- [x] Fourier Neural Operator 구현
- [x] TCAD Dataset 생성
- [x] TCAD Surrogate Model 개발
- [ ] 다양한 Boundary Condition 일반화 성능 비교
- [ ] PINN vs DeepONet vs FNO 성능 비교

---

### Paper Link

https://arxiv.org/abs/2006.01634