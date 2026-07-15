# Neural Fields for Device Simulation

> Anonymous et al. (2024)
>
> **Neural Fields for Device Simulation**
>
> arXiv Preprint

---

## Why I Read This Paper

기존 TCAD AI 모델들은 대부분 Mesh 또는 Grid 위에서 정의된 물리량을 예측한다.

하지만 실제 반도체 소자는

- 매우 복잡한 Geometry
- 서로 다른 Mesh Resolution
- 다양한 Boundary Condition

을 갖기 때문에 Grid 기반 Neural Operator나 Graph Network만으로 일반화하기 어렵다.

이 논문은 이러한 한계를 해결하기 위해 **Neural Field(Implicit Neural Representation)** 를 Device Simulation에 적용한다.

좌표를 입력하면 해당 위치의 물리량을 직접 예측하는 방식으로, Mesh에 의존하지 않는 새로운 형태의 TCAD Surrogate Model을 제안한다.

---

## Problem

기존 AI 기반 TCAD 모델은 대부분

```
Mesh

↓

Node Prediction
```

방식이다.

하지만

- Mesh Resolution이 변경되면 성능이 저하된다.
- Device Geometry가 달라지면 재학습이 필요하다.
- 다양한 구조에 일반화하기 어렵다.

즉,

Simulation Domain 자체를 표현하지 못한다.

---

## Key Idea

핵심 아이디어는

**Device를 Neural Field로 표현하는 것이다.**

기존 방식

```
Mesh

↓

Potential
```

Neural Field

```
(x,y,z)

↓

MLP

↓

Potential
```

즉,

좌표를 입력하면

해당 위치의

- Potential

- Carrier Density

- Electric Field

- Current Density

등을 예측한다.

Simulation 자체를 하나의 연속 함수로 학습한다.

---

## Method

### Step 1. Coordinate Encoding

입력은

```
(x,y,z)
```

이다.

여기에

Positional Encoding을 적용한다.

```
Coordinate

↓

Fourier Feature

↓

Encoded Coordinate
```

고주파 정보를 잘 표현할 수 있도록 한다.

---

### Step 2. Neural Field

MLP가

```
Coordinate

↓

Physics Quantity
```

를 학습한다.

예를 들어

```
(0.13,0.45)

↓

Potential
```

또는

```
Carrier Density
```

를 출력한다.

---

### Step 3. Multi-Physics Learning

하나의 Network가 동시에

- Potential
- Electron Density
- Hole Density
- Electric Field

등을 예측한다.

즉

Multi-task Learning 형태이다.

---

### Step 4. Continuous Query

Mesh Node뿐 아니라

임의의 위치도 바로 예측 가능하다.

```
Coordinate

↓

Prediction
```

필요한 Resolution으로 자유롭게 Sampling할 수 있다.

---

## Equation

Neural Field는

좌표

\[
x=(x,y,z)
\]

를 입력받아

\[
f_\theta(x)
\]

를 계산한다.

출력은

\[
f_\theta(x)
=
(V,n,p,E)
\]

이다.

Loss는

\[
L
=
L_{Potential}
+
L_{Carrier}
+
L_{Electric}
\]

등 여러 물리량을 동시에 최소화한다.

필요 시 PDE Residual을 추가하여 Physics Constraint를 적용할 수도 있다.

---

## Advantages

- Mesh Resolution에 독립적
- Continuous Representation 가능
- 다양한 Geometry 일반화
- Arbitrary Coordinate Query 가능
- Compact Memory
- TCAD Surrogate에 적합
- PINN과 쉽게 결합 가능

---

## Limitations

- 매우 복잡한 Device에서는 학습 시간이 길다.
- Coordinate Encoding 방식에 성능이 민감하다.
- Boundary Condition 처리가 어렵다.
- 매우 큰 Device에서는 MLP 용량이 증가한다.
- 아직 산업 적용 사례는 제한적이다.

---

## Key Takeaways

- Device를 Mesh가 아닌 함수(Function)로 표현한다.
- 좌표를 입력하면 원하는 물리량을 바로 예측한다.
- Neural Field는 Continuous TCAD Representation이다.
- 향후 Implicit TCAD Solver의 핵심 기술이 될 가능성이 높다.

---

## Personal Notes

### Relation to My Current Project

현재 GitHub 프로젝트는

```
GCA Equation

↓

PINN
```

이다.

향후에는

```
PINN

↓

Neural Field

↓

TCAD Surrogate
```

순으로 확장하고 싶다.

---

### Relation to My Career

현재 업무에서도

공정 조건

↓

소자 특성

↓

보상 알고리즘

을 모델링하였다.

Neural Field 역시

좌표

↓

물리량

을 모델링한다는 점에서

물리 현상을 함수로 표현한다는 철학이 동일하다.

---

### Relation to SK hynix TCAD

SK하이닉스 JD의

- Device Simulation
- Physics AI
- Scientific Machine Learning

과 매우 잘 연결된다.

특히

Future TCAD에서는

```
Coordinate

↓

Neural Field

↓

TCAD Solver
```

형태가 중요한 연구 주제가 될 것으로 예상된다.

---

### Comparison with Previous Papers

| Paper | Representation | 특징 |
|---------|----------------|----------------|
| PINN | PDE Solution | Physics Loss |
| DeepONet | Operator | Function Mapping |
| FNO | Fourier Space | Fast Operator |
| MeshGraphNet | Mesh | Graph Simulation |
| UPT | Universal Latent | Foundation Model |
| **Neural Fields** | Coordinate | Continuous Device Representation |

Neural Field는 Device 자체를 하나의 연속 함수로 표현한다는 점에서 기존 방법들과 차별화된다.

---

### Relation to TCAD

향후 AI 기반 TCAD는

```
Material

↓

MLIP

↓

TCAD

↓

Neural Field

↓

Neural Operator

↓

Foundation Model
```

순으로 발전할 가능성이 있다.

Neural Field는 Device를 연속 함수로 표현하는 핵심 기술이다.

---

### Future Learning Plan

이 논문를 기반으로 다음 프로젝트를 진행할 계획이다.

- [ ] Coordinate-based MLP 구현
- [ ] Fourier Feature Encoding 구현
- [ ] Implicit Neural Representation 학습
- [ ] TCAD Potential Prediction
- [ ] PINN + Neural Field 결합
- [ ] Continuous Device Surrogate 개발

---

### Paper Link

https://arxiv.org/search/?query=Neural+Fields+for+Device+Simulation&searchtype=all