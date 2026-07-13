# Machine Learning Interatomic Potentials (MLIP)

> Ryan Jacobs et al. (2025)
>
> **A Practical Guide to Machine Learning Interatomic Potentials – Status and Future**
>
> Current Opinion in Solid State and Materials Science (2025)

---

## Why I Read This Paper

최근 SK하이닉스 TCAD JD에서 **Material Simulation**, **Physics AI**, **Machine Learning Interatomic Potentials (MLIP)**를 중요한 기술 키워드로 제시하고 있습니다.

기존에는 DFT(Density Functional Theory)를 이용하여 원자 수준(Material Level)의 물성을 계산하고, 이를 TCAD로 전달하는 방식이 일반적이었습니다.

하지만 DFT는 계산 비용이 매우 커서 수많은 후보 물질을 탐색하기 어렵습니다.

MLIP는 DFT 수준의 정확도를 유지하면서 계산 속도를 수천~수만 배 향상시켜 Material Discovery와 Virtual R&D를 가능하게 하는 핵심 기술입니다.

PINN, Neural Operator가 **Device Physics AI**라면, MLIP는 **Atomistic Physics AI**라고 볼 수 있습니다. :contentReference[oaicite:0]{index=0}

---

## Problem

반도체 Material Simulation은 일반적으로

```
Atomic Structure

↓

DFT

↓

Energy

↓

Force

↓

Material Property
```

순서로 계산됩니다.

그러나 DFT는

- 계산 시간이 매우 길고
- 원자 수가 증가하면 계산량이 급격히 증가하며
- 대규모 Molecular Dynamics(MD)에 적용하기 어렵습니다.

반면 Classical Force Field는

- 계산은 매우 빠르지만
- 새로운 소재나 화학 결합을 정확하게 표현하지 못합니다.

즉,

```
DFT
=
정확하지만 느림

Classical Potential
=
빠르지만 부정확
```

이라는 Accuracy-Speed Trade-off가 존재합니다. :contentReference[oaicite:1]{index=1}

---

## Key Idea

MLIP의 핵심은

**DFT가 계산하는 원자 간 상호작용(Interatomic Potential)을 AI가 학습하는 것**입니다.

기존

```
Atomic Structure

↓

DFT

↓

Energy
```

제안 방식

```
Atomic Structure

↓

Machine Learning

↓

Energy
```

뿐만 아니라

```
Atomic Structure

↓

Machine Learning

↓

Energy

↓

Atomic Force
```

까지 동시에 예측합니다.

즉,

DFT Solver 자체를 AI Surrogate Model로 대체하는 것입니다.

---

## Method

### Step 1. DFT Dataset 생성

먼저 다양한 원자 구조에 대해 DFT 계산을 수행합니다.

생성되는 데이터는

- Total Energy
- Atomic Force
- Stress Tensor
- Atomic Position

등입니다.

---

### Step 2. Atomic Representation

원자 구조를

AI가 이해할 수 있는 형태로 변환합니다.

대표적인 Descriptor

- SOAP
- ACSF
- ACE
- Graph Representation

최근에는

Descriptor 없이

End-to-End GNN을 사용하는 경우도 많습니다.

---

### Step 3. Machine Learning

AI는

```
Atomic Position

↓

Neural Network

↓

Energy
```

를 학습합니다.

Energy를 미분하면

Atomic Force를 얻을 수 있습니다.

\[
F_i
=
-\frac{\partial E}{\partial r_i}
\]

따라서

Energy만 정확히 학습하면

Force도 자동으로 계산됩니다.

---

### Step 4. Molecular Dynamics

학습된 MLIP를

LAMMPS

ASE

Molecular Dynamics

등과 연결하여

실제 원자 시뮬레이션을 수행합니다.

---

## Equation

원자계의 총 에너지는

\[
E
=
\sum_iE_i
\]

로 표현됩니다.

각 원자의 Local Environment를 이용하여

\[
E_i=f(x_i)
\]

를 학습합니다.

Atomic Force는

\[
F_i
=
-\nabla_iE
\]

입니다.

Loss는

Energy와 Force를 동시에 사용합니다.

\[
L
=
\lambda_EL_E
+
\lambda_FL_F
\]

여기서

- Energy Loss
- Force Loss

를 함께 최소화합니다.

---

## Advantages

- DFT 대비 수천~수만 배 빠른 계산
- DFT 수준의 높은 정확도
- 대규모 Molecular Dynamics 가능
- Material Screening 가능
- New Material Discovery 가능
- TCAD Material Parameter 생성 가능
- Virtual R&D 핵심 기술

최근에는 범용 Foundation MLIP도 등장하여 여러 재료군에 대한 활용 가능성이 확대되고 있습니다. :contentReference[oaicite:2]{index=2}

---

## Limitations

- DFT 데이터 생성 비용이 매우 크다.
- 학습 데이터 밖에서는 정확도가 떨어질 수 있다.
- 새로운 화학종에 대한 일반화가 어렵다.
- Long-range Interaction 처리가 쉽지 않다.
- Magnetic Material이나 Excited State는 아직 어려운 분야이다.
- 학습 데이터 품질이 성능을 크게 좌우한다. :contentReference[oaicite:3]{index=3}

---

## Key Takeaways

- MLIP는 DFT를 AI로 근사하는 기술이다.
- Energy와 Force를 동시에 학습한다.
- Material Discovery 속도를 획기적으로 높인다.
- 차세대 반도체 Material Simulation의 핵심 기술이다.
- Physics AI와 Virtual R&D의 기반 기술 중 하나이다.

---

## Personal Notes

### Relation to My Current Project

현재 진행 중인 GitHub 프로젝트는

```
Physics Equation

↓

PINN

↓

Device Prediction
```

이다.

앞으로는

```
Atomic Structure

↓

MLIP

↓

Material Property

↓

TCAD

↓

PINN

↓

Device Prediction
```

까지 확장해보고 싶다.

---

### Relation to My Career

MLIP는

```
원자 구조

↓

AI

↓

Material Property
```

를 수행한다.

즉,

내 커리어를

**Device Physics**

에서

**Atomistic Physics**

로 확장하는 첫 번째 기술이다.

---

### Relation to SK hynix TCAD

SK하이닉스 JD에서 언급한

- Material Simulation
- AI Physics
- Virtual R&D

의 가장 앞단에 위치하는 기술이다.

전체 Workflow는

```
Atomic Structure

↓

MLIP

↓

Material Property

↓

TCAD

↓

Device Characteristics

↓

Circuit

↓

System
```

이다.

즉,

MLIP는

반도체 Physics AI의 시작점이다.

---

### Comparison with Previous Papers

| Paper | Physics Scale | AI 역할 |
|---------|----------------|------------------------------|
| PINN | Device | PDE Solution Learning |
| DeepONet | Device | Operator Learning |
| Fourier Neural Operator | Device | Fast PDE Solver |
| MeshGraphNet | Device | Mesh Simulation |
| Physics-Informed GNN | Device | Graph + Physics |
| **MLIP** | **Atomic** | **DFT Surrogate** |

MLIP는 지금까지 리뷰한 논문들과 달리

**Device Level이 아닌 Material Level**을 다룬다는 것이 가장 큰 차이점이다.

---

### Relation to TCAD

향후 AI 기반 반도체 연구는

```
DFT

↓

MLIP

↓

Molecular Dynamics

↓

Material Parameter

↓

TCAD

↓

Surrogate TCAD

↓

Circuit Simulation
```

형태로 연결될 가능성이 높다.

즉,

MLIP는

TCAD보다 한 단계 아래(Material Layer)의 Physics AI이다.

---

### Future Learning Plan

이 논문를 기반으로 다음 프로젝트를 진행할 계획이다.

- [ ] DFT 기본 원리 학습
- [ ] Molecular Dynamics 학습
- [ ] LAMMPS 사용법 익히기
- [ ] ASE(Atomic Simulation Environment) 학습
- [ ] MACE, NequIP, CHGNet 등 최신 MLIP 모델 비교
- [ ] 간단한 MLIP 추론 실습
- [ ] MLIP → TCAD → PINN으로 이어지는 Physics AI Pipeline 구축

---

### Paper Link

https://doi.org/10.1016/j.cossms.2025.101214