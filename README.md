# Physics-AI-Lab

> Understanding Physics.
> Building Algorithms.
> Accelerating Research with AI.

An open research portfolio bridging Physics-based Algorithm Engineering and AI-driven semiconductor device / material simulation.

물리 현상을 이해하고, 모델링하고, AI로 구현해 실제 시스템에 적용하는 과정을 기록하는 개인 연구 저장소입니다. 9년간 OLED 디스플레이 소자의 물성 기반 보상 알고리즘을 개발해온 경험을, Physics-Informed AI와 반도체 소자/물성 시뮬레이션 영역으로 확장하고 있습니다.

---

## Vision

반도체 R&D는 전통적인 시뮬레이션을 넘어, 물리·수치해석·AI가 융합된 방향으로 진화하고 있습니다. 이 저장소의 목표는 이 흐름을 이루는 핵심 기술들(PINN, Neural Operator, TCAD, MLIP, NEGF, Monte Carlo)을 직접 학습하고 구현하며 이해하는 것입니다.

---

## Research Areas

- Physics-informed Neural Networks (PINNs)
- Neural Operators (FNO, DeepONet)
- TCAD (Device / Process Simulation)
- Machine Learning Interatomic Potentials (MLIP)
- NEGF / Quantum Transport
- Monte Carlo Simulation (Ion Implantation)
- Graph Neural Networks for Mesh-Based Simulation (MeshGraphNet)
- Scientific Machine Learning
- Numerical Methods (FEM / FVM)

---

## Repository Structure

```
docs/            연구 로드맵 및 참고자료
paper-reviews/   논문 리뷰 (paper-review-template.md는 리뷰 템플릿)
projects/        구현 프로젝트 (01~07, 번호는 진행 순서)
notes/           물리·수치해석 개념 정리
```

---

## Projects

### 01. GCA-based TFT Id-Vg PINN
TFT 소자의 Gradual Channel Approximation 방정식을 PINN으로 학습. 데이터가 희소한 영역에서 물리 제약이 얼마나 도움이 되는지 baseline과 정량 비교.

- 데이터 부족 상황(150/5000 샘플)에서 물리 제약 모델이 baseline 대비 오차 개선
- GCA 방정식 자체의 경계 불연속을 발견하고 continuity correction으로 해결 → 불연속을 없애자 PINN의 이점이 14.9%→5.2%로 줄어드는 것을 확인 (물리 제약은 데이터로 배우기 어려운 영역에서 가장 유용하다는 시사점)

→ [`projects/01_gca-pinn`](./projects/01_gca-pinn)

### 02. Neural Operator — Vth Generalization (PhysicsNeMo FNO)
Project 01의 점 단위 PINN(고정 소자)을 넘어, 소자 파라미터(Vth)가 달라져도 재학습 없이 대응하는 Operator를 NVIDIA PhysicsNeMo의 공식 FNO로 학습.

- 학습에 없던 Vth 값에 대한 held-out MSE 0.000052로 일반화 확인

→ [`projects/02_neural-operator`](./projects/02_neural-operator)

### 03. TCAD Surrogate — Self-built 1D Device Simulator
Sentaurus 라이선스 없이 직접 구현한 1D Poisson / Poisson-Drift-Diffusion 소자 시뮬레이터. Box-integration(FVM)/FEM + Newton-Raphson, Scharfetter-Gummel discretization + Gummel iteration 등 실제 TCAD가 쓰는 수치기법을 그대로 구현.

- MOS 커패시터 quasi-static C-V 곡선(축적-공핍-반전) 검증
- PN 다이오드 Shockley I-V 법칙을 18자리(order of magnitude)에 걸쳐 검증
- 자체 생성 데이터로 FNO 서로게이트 학습 → 657배 가속 (435ms → 0.66ms/curve)

→ [`projects/03_tcad-surrogate`](./projects/03_tcad-surrogate)

### 04. Toy MLIP — Neural Network Potential for LJ Clusters
Lennard-Jones 포텐셜 기반 MD로 생성한 데이터를 Behler-Parrinello 스타일 신경망 포텐셜(NNP)로 학습. Material Simulation(MD) 영역의 첫 구현.

- 미분가능한 radial symmetry function descriptor 직접 구현
- Energy + Force dual loss로 학습, 학습에 없던 온도(T=0.8)에서 에너지 RMSE 0.79로 일반화
- Force-matching이 energy-matching보다 어렵다는 실제 MLIP 문헌의 알려진 난제를 직접 재현·기록

→ [`projects/04_mlip`](./projects/04_mlip)

### 05. NEGF Quantum Transport — 1D Tight-Binding Toy Model
반무한 리드에 결합된 1D tight-binding 체인의 양자수송을 NEGF로 계산.

- 균일 체인 T(E)=1 검증 (이론값 대비 오차 1e-6)
- Double-barrier 공명터널링 구조에서 공명 피크 재현
- Landauer 공식으로 I-V 특성 계산, RTD의 핵심 특징인 NDR(음의 미분저항) 재현

→ [`projects/05_negf`](./projects/05_negf)

### 06. Ion Implantation Monte Carlo (Simplified BCA)
TCAD의 Process Simulation 영역 — Boron 이온을 Silicon 타겟에 주입하는 이온주입 공정을 단순화된 Binary Collision Approximation(BCA) Monte Carlo로 시뮬레이션. 산란각은 2체 탄성충돌의 운동학적으로 정확한 관계식으로, 에너지 전달은 LSS 이론 유도에 쓰이는 표준 power-law 미분단면적 근사로 샘플링.

- 투사 범위 Rp가 에너지에 따라 단조증가, power-law 지수(E^0.68)가 LSS 이론 예측 범위(0.6~1.0) 안에 정확히 들어옴
- Straggling 비율(ΔRp/Rp≈0.23~0.29), 절대 Rp 스케일 모두 실제 문헌값과 같은 자릿수로 검증
- 초기 numpy 벡터 구현이 병목이 되어(이온 100개에 3.8초), 순수 파이썬 float 연산으로 재작성해 20배 가속

→ [`projects/06_mc-implant`](./projects/06_mc-implant)

### 07. MeshGraphNet — Curvature-Driven Front Evolution (Etch Profile Toy Model)
[SK하이닉스 TCAD Intelligence 팀 블로그](./paper-reviews/00_SK-hynix-NVIDIA-AI-Physics-for-TCAD.md)와 [논문 리뷰 12(MeshGraphNets)](./paper-reviews/12_MeshGraphNets.md)에서 다룬 GNN 기반 mesh 시뮬레이션을, 곡률 흐름으로 매끄러워지는 식각 경계면이라는 단순화된 문제로 직접 구현. Encode-Process-Decode 아키텍처(Pfaff et al., 2020)를 PyTorch로 직접 구현 (DGL 의존성 회피).

- Ground truth 시뮬레이터(front-tracking)를 반원 수축의 해석해와 비교해 먼저 검증 (상대오차 2.1%)
- One-step 예측은 거의 완벽(MSE ≈3×10⁻⁸)하지만, 80스텝 자기회귀 rollout에서는 오차가 2233배 누적 — 실제 MeshGraphNet 문헌에 알려진 난제를 직접 재현, SK하이닉스 블로그가 언급한 "매 iteration 재메싱 필요성"과 연결지어 해석

→ [`projects/07_meshgraphnet`](./projects/07_meshgraphnet)

---

## Paper Reviews

`paper-reviews/`에 논문을 읽고 정리 중입니다 (`docs/paper-review-template.md`는 리뷰 템플릿).

| # | Title |
|---|---|
| 00 | (SK hynix + NVIDIA) Using AI Physics for Technology Computer-Aided Design Simulations |
| 01 | Physics-Informed Neural Networks (PINNs) |
| 02 | DeepONet |
| 03 | Fourier Neural Operator (FNO) |
| 04 | Physics-informed AI Accelerated Retention Analysis of Ferroelectric Vertical NAND |
| 05 | Revolutionizing TCAD Simulations with Universal Device Encoding and Graph Attention Networks |
| 06 | Physics-Informed Neural Network for Predicting Out-of-Training-Range TCAD Solution with Minimized Domain Expertise |
| 07 | FlashTP |
| 08 | S-ANN: Synchronous TCAD Device Simulation of FinFET using Artificial Neural Network |
| 09 | Physics-Informed Neural Networks for Device and Circuit Modeling: A Case Study of NeuroSPICE |
| 10 | Automated, Physics-Guided AI Framework for Asymmetry-Aware Ferroelectric Compact Models |
| 11 | Neural Operators as Data-Driven Surrogate Models for Partial Differential Equations |
| 12 | MeshGraphNets |
| 13 | Physics-Informed Graph Neural Networks |
| 14 | Machine Learning Interatomic Potentials (MLIP) |
| 15 | Universal Physics Transformers (UPT) |
| 16 | Neural Fields for Device Simulation |
| 17 | Graph Neural Networks for Semiconductor Device Modeling |
| 18 | Physics-Embedded Neural Operator |
| 19 | AI-Driven Multi-Physics Simulation for Semiconductor Devices |
| 20 | AI for Electronic Design Automation A Survey |
| 21 | NVIDIA PhysicsNeMo |

---

## Next Steps

- [ ] PINN for 1D Heat / Poisson Equation (기초 PDE로 범위 확장)
- [ ] NEGF I-V 곡선 노이즈 개선 (격자 해상도/적응형 적분)
- [ ] MLIP force-matching 정확도 개선 (각도 항 descriptor 추가)
- [ ] AD-NEGF, DeePTB-NEGF 리뷰 후 ML 기반 NEGF 가속 실험
- [ ] Ion Implantation MC를 ZBL magic formula 기반 정밀 모델로 확장, channeling 효과 추가
- [ ] MeshGraphNet rollout 안정성 개선 (re-meshing, noise injection 등 Pfaff et al. 기법 적용) 및 2D 삼각형 mesh로 확장

---

## Long-term Goal

```
Physics
   ↓
Simulation
   ↓
Machine Learning
   ↓
Semiconductor Virtual R&D
```

---

## Disclaimer

This repository is an independent personal learning project.
The code and documents are implemented solely for educational and research purposes.
