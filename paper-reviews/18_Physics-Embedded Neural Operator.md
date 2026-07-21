# Physics-Embedded Neural Operator (Boundary-Embedded Neural Operators)

> Wang et al. (2024)
>
> Boundary-Embedded Neural Operators for Elliptic PDEs
>
> ICLR 2024

---

## Why I Read This Paper

Neural Operator(FNO, DeepONet)는 PDE Solution Operator를 매우 빠르게 근사할 수 있지만, 실제 공학 문제에서는 복잡한 경계조건(Boundary Condition) 때문에 정확도가 크게 저하되는 경우가 많다.

특히 TCAD에서는

- Device Geometry
- Contact
- Dirichlet Boundary
- Neumann Boundary
- Material Interface

등이 모두 경계조건으로 작용한다.

이 논문은 이러한 경계조건을 Loss Function이 아닌 Network Architecture 자체에 포함시키는 방법을 제안한다.

---

## Problem

기존 Neural Operator는

```
Input

↓

Neural Operator

↓

Solution
```

형태이다.

하지만 실제 PDE는

```
PDE

+

Boundary Condition
```

을 동시에 만족해야 한다.

기존 방법은

Boundary Loss를 추가하여

```
Loss

=

Data Loss

+

Physics Loss

+

Boundary Loss
```

를 최소화한다.

그러나

- Boundary가 복잡할수록 학습이 어려워지고
- Geometry가 변하면 일반화 성능이 감소한다.

---

## Key Idea

Boundary Condition 자체를

Neural Operator의 입력 구조 안으로 직접 포함시킨다.

즉,

기존

```
Input

↓

Operator

↓

Output
```

에서

```
Input

+

Boundary Embedding

↓

Operator

↓

Output
```

으로 변경한다.

Boundary 정보가 Feature로 함께 전달되므로

별도의 Boundary Loss에만 의존하지 않고

Physics를 자연스럽게 학습한다.

---

## Method

### Step 1. PDE 입력 구성

입력에는

- Initial Field
- Material Property
- Source Term

뿐 아니라

Boundary Condition도 함께 포함한다.

---

### Step 2. Boundary Embedding

Boundary를

Feature Space로 변환한다.

예를 들어

```
Boundary Mask

↓

Embedding Layer

↓

Boundary Feature
```

를 생성한다.

---

### Step 3. Neural Operator

Boundary Feature와

Physical Field를 동시에 입력받아

Operator Learning을 수행한다.

주로

- Fourier Layer
- Spectral Convolution

등이 사용된다.

---

### Step 4. PDE Solution

Boundary 정보를 고려한

Solution Operator를 학습한다.

복잡한 Geometry에서도

보다 안정적인 해를 예측할 수 있다.

---

## Equation

Operator Learning은

\[
u=\mathcal{G}(a)
\]

를 학습한다.

Boundary 정보를 포함하면

\[
u=\mathcal{G}(a,b)
\]

가 된다.

여기서

- \(a\) : PDE 입력
- \(b\) : Boundary Condition

Loss는

\[
L=L_{data}+L_{physics}
\]

를 사용하지만,

Boundary 정보는

Loss가 아니라

입력 Feature에 직접 포함된다. :contentReference[oaicite:3]{index=3}

---

## Advantages

- Boundary Condition을 직접 학습
- 복잡한 Geometry 처리 가능
- 기존 PINO보다 높은 일반화 성능
- Mesh 변화에 강인함
- 실제 공학 문제 적용성이 높음
- TCAD와 같은 Device Simulation에 적합

---

## Limitations

- Elliptic PDE 중심으로 검증되었다.
- Time-dependent PDE에는 추가 연구가 필요하다.
- 매우 복잡한 3D Device에서는 메모리 요구량이 증가한다.
- 학습 데이터 품질에 영향을 받는다.

---

## Key Takeaways

- Boundary Condition을 Loss가 아니라 입력 구조에 포함한다.
- Physics를 Network Architecture 수준에서 내재화한다.
- 실제 Engineering PDE에서 기존 FNO보다 높은 일반화 성능을 보인다.
- TCAD와 같은 복잡한 Device Simulation에 적합한 구조이다.

---

## Personal Notes

### Relation to My Current Project

현재 프로젝트는

```
GCA Equation

↓

PINN
```

구조이다.

향후에는

```
PINN

↓

PINO

↓

Boundary-Embedded Neural Operator

↓

TCAD Surrogate
```

순으로 확장하고 싶다.

---

### Relation to SK hynix TCAD

반도체 TCAD에서는

- Gate
- Source
- Drain
- Oxide Interface
- Material Boundary

모두 Boundary Condition이다.

Boundary를 Network 내부에 직접 표현하는 방식은

실제 Device Simulation과 매우 잘 맞는다.

특히 향후

- Sentaurus TCAD
- Process Simulation
- Device Simulation

Surrogate를 구축할 때 매우 중요한 아이디어가 될 수 있다.

---

### Comparison with Previous Papers

| Paper | Physics 반영 방식 |
|---------|----------------|
| PINN | PDE Loss |
| PINO | PDE Residual + Operator Learning |
| DeepONet | Operator Learning |
| FNO | Fourier Operator |
| **Boundary-Embedded Neural Operator** | Boundary를 Network 구조에 직접 포함 |
| Universal Physics Transformer | 다양한 Physics를 하나의 Foundation Model로 통합 |

---

### Why This Paper Matters

PINN은

"Physics를 Loss에 넣는다."

PINO는

"Physics를 Operator에 넣는다."

Boundary-Embedded Neural Operator는

"Boundary 자체를 Network 구조 안에 넣는다."

즉,

Physics를 점점 더 Network Architecture 수준으로 내재화하는 방향으로 발전하고 있음을 보여주는 중요한 논문이다.

---

### Future Learning Plan

- [ ] PINO 구현
- [ ] Boundary Encoding 구현
- [ ] TCAD Boundary Representation 설계
- [ ] FNO + Boundary Embedding 실험
- [ ] PINO와 Boundary-Embedded 모델 성능 비교
- [ ] TCAD Surrogate에 적용

---

### Paper Link

https://proceedings.iclr.cc/paper_files/paper/2024/file/218ca0d92e6ed8f9db00621e103dc70c-Paper-Conference.pdf