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
- Compact Device Modeling / Parameter Extraction
- Scientific Machine Learning
- Numerical Methods (FEM / FVM)

---

## Repository Structure

```
docs/            연구 로드맵 및 참고자료
paper-reviews/   논문 리뷰 (paper-review-template.md는 리뷰 템플릿)
projects/        구현 프로젝트 (01~13, 번호는 진행 순서)
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

→ [`projects/06_mc`](./projects/06_mc)

### 07. MeshGraphNet — Curvature-Driven Front Evolution (Etch Profile Toy Model)
[SK하이닉스 TCAD Intelligence 팀 블로그](./paper-reviews/00_SK-hynix-NVIDIA-AI-Physics-for-TCAD.md)와 [논문 리뷰 12(MeshGraphNets)](./paper-reviews/12_MeshGraphNets.md)에서 다룬 GNN 기반 mesh 시뮬레이션을, 곡률 흐름으로 매끄러워지는 식각 경계면이라는 단순화된 문제로 직접 구현. Encode-Process-Decode 아키텍처(Pfaff et al., 2020)를 PyTorch로 직접 구현 (DGL 의존성 회피).

- Ground truth 시뮬레이터(front-tracking)를 반원 수축의 해석해와 비교해 먼저 검증 (상대오차 2.1%)
- One-step 예측은 거의 완벽(MSE ≈3×10⁻⁸)하지만, 80스텝 자기회귀 rollout에서는 오차가 2233배 누적 — 실제 MeshGraphNet 문헌에 알려진 난제를 직접 재현, SK하이닉스 블로그가 언급한 "매 iteration 재메싱 필요성"과 연결지어 해석

→ [`projects/07_meshgraphnet`](./projects/07_meshgraphnet)

### 08. Self-Consistent NEGF-Poisson — Quantum MOS Capacitor (Capstone)
[프로젝트 03(TCAD Surrogate)](./projects/03_tcad-surrogate)과 [프로젝트 05(NEGF)](./projects/05_negf)를 결합한 capstone. MOS 반전층 전자를 classical Boltzmann 대신 NEGF로 계산해, "quantum confinement로 인한 전하 setback" 재현을 시도. 다른 프로젝트들과 달리 이번엔 **디버깅 여정 자체가 핵심 스토리**.

- 단위 버그(nm→m 변환 누락으로 결합항이 10⁹배 작아짐) 발견 및 수정
- 격자 스케일 진동을 "버그인지 실제 물리(Friedel 진동 유사 정상파 간섭)인지" 단일 에너지 국소 상태밀도로 직접 진단해 구분
- 1D 체인의 2D in-plane 상태밀도 누락이라는 근본적 한계를 정직하게 기록 (bulk-anchoring calibration으로 우회, 정량적 한계 명시)
- Self-consistent damped fixed-point 루프는 안정적으로 수렴 확인

→ [`projects/08_negf-poisson`](./projects/08_negf-poisson)

### 09. Bayesian Optimization for TCAD Device Design
[논문 리뷰 10(Ferroelectric Compact Models, active learning으로 TCAD 호출 최소화)](./paper-reviews/10_Automated,%20Physics-Guided%20AI%20Framework%20for%20Asymmetry-Aware%20Ferroelectric%20Compact%20Models.md)에서 다룬 기법을 프로젝트 03 스타일의 MOS 커패시터 시뮬레이터에 직접 적용. "목표 문턱전압을 만족하는 (산화막 두께, 도핑농도)를 최소 시뮬레이션 횟수로 찾기". Gaussian Process 회귀 + Expected Improvement를 scikit-learn 없이 직접 구현.

- Grid resolution 문제(프로젝트 08과 같은 종류)로 Vth 정의를 C-V 최솟값 기준으로 수정
- 이산 격자값을 그대로 반환해 목표함수가 계단함수가 되는 버그를 포물선 보간으로 해결 — 이후 BO와 Random Search의 성능 차이가 명확하게 드러남
- 5 seed 평균 비교 결과, **BO가 Random Search 대비 최종 오차 9.4배 낮음**, 편차도 훨씬 작아 일관되게 좋은 해를 찾음

→ [`projects/09_bo_tcad`](./projects/09_bo_tcad)

### 10. 1D Kohn-Sham DFT Solver (Soft-Coulomb Potential)
3D 쿨롱 특이점을 피하는 1D DFT 문헌의 표준 관행(soft-Coulomb 포텐셜)으로 단일 전자 및 2전자 self-consistent Kohn-Sham 계산을 구현.

- Milestone 1(단일 전자 soft-Coulomb): 문헌 벤치마크 기저상태 에너지(-0.6698 Hartree)와 **소수점 4자리까지 정확히 일치**
- Milestone 2(2전자, Hartree+exchange SCF): 밀도 정규화(∫n=2.0)를 유지하며 안정적으로 수렴, exchange 포함 시 에너지가 낮아지고 밀도가 응집되는 물리적으로 타당한 방향 확인
- 한계를 정직하게 기록: exchange-correlation은 정확한 1D LDA(QMC 피팅)가 아닌 단순화된 선형 근사로, SCF 방법론과 정성적 경향 검증에 집중하고 정량적 벤치마크 재현은 주장하지 않음

→ [`projects/10_dft`](./projects/10_dft)

### 11. wafer2device — 웨이퍼 패턴에서 소자 특성 보상까지 (물리 검증 게이트)
웨이퍼 불량 패턴에서 공정 변동 가설을 세우고 소자 특성 보상까지 이어지는 파이프라인. 자율성을 한 곳에만 두고, 나머지는 결정론적으로 고정.

- 단계 순서는 고정 DAG, 물리 검증 실패 시에만 구조화된 피드백으로 가설을 재수립하는 루프(최대 3회)
- 검증 기준과 허용 파라미터를 설정 파일로 고정해 모델이 바꿀 수 없게 하고, 끝내 통과하지 못하면 사람에게 넘기며 이후 단계를 실행하지 않음
- 모든 값에 출처 태그(measured/assumed/predicted/optimized)를 붙여 실데이터와 물리 가정을 코드 레벨에서 분리
- 규칙 기반 기준선의 결함을 두 번 고치자 AI의 이득이 절반으로 줄어든 결과를 그대로 기록. 단순 패턴에서는 이득이 없고 복합 패턴에서만 유효

→ [`projects/11_wafer2device`](./projects/11_wafer2device)

### 12. Compact Model Parameter Extraction — 곡선은 맞는데 파라미터를 믿을 수 있는가
7-파라미터 Compact Model(EKV 형태 자체 구현, BSIM 계열 아님)을 만들고, 그 파라미터 추출을 자동화한 뒤 **식별성(identifiability)** 을 정량화.

- Subthreshold → DIBL → Linear → Saturation 순으로 단계 추출하고, 각 단계마다 물리 타당성 게이트(열적 하한, 물리 범위, 경계 고착 검출)를 통과해야 다음 단계로 진행
- Jacobian 기반 공분산 C = σ²(JᵀJ)⁻¹로 파라미터별 1σ와 상관을 계산. 예측한 1σ를 노이즈 실현을 바꿔 200회 반복 추출한 분포와 비교해 **7개 중 6개가 10% 이내로 일치**, `ss` 하나는 예측이 1.54배 낙관적 — 반복 측정 전에 어떤 파라미터를 신뢰할지 판단 가능. 어긋난 하나의 원인(`measure()`에는 노이즈 플로어가 있는데 `analyse()`는 균일 노이즈를 가정)은 [프로젝트 13 Stage 1](./projects/13_measurement-to-circuit/stage1_design)에서 규명
- 민감도가 파라미터 간 100배 차이남(theta는 vth0의 약 1/100). vth0↔mu0 상관 +0.93은 선형 영역에서 Id ∝ μ(Vgs−Vth)이므로 물리적으로 예측되는 degeneracy
- 단계별 추출은 15/15 에스컬레이션, 동시 fitting은 0/15. **에스컬레이션한 쪽이 옳음** — 동시 fitting은 5% 노이즈에서 17%, 10%에서 33% 틀린 theta를 똑같이 자신 있게 돌려줌
- 한계 명시: BSIM/BSIM-CMG 구현이 아니고, 단일 geometry라 theta와 rs의 분리는 원리적으로 불가(여러 채널 길이 필요). 모델 불일치가 없는 합성 데이터

→ [`projects/12_compact-model-extraction`](./projects/12_compact-model-extraction)

### 13. Measurement → Circuit — Compact Model이 쓸 만하려면 무엇이 참이어야 하는가
프로젝트 12의 모델이 실제로 쓰이기까지 거치는 **세 인계 지점**을 각각 검증. 측정 설계 → 코너·통계 모델 → 회로 수준. 세 단계를 독립 릴리즈하고, 뒤 단계는 앞 단계의 결과를 다시 만들지 않고 그대로 가져다 씀. 각 단계마다 측정 전에 예측을 등록하고 **빗나간 것을 그대로 기록**.

- **Stage 1 (측정 설계)** — 같은 12회 측정에서 정보량 기준으로 점을 고르니 held-out 예측 오차 **2.9배** 개선. 그런데 **교과서 D-optimal이 더 나쁨**: 균일 노이즈를 가정해 민감도가 가장 큰 깊은 Subthreshold에 점을 몰아넣는데 거기가 전류가 계측 노이즈 플로어에 가까워 상대 정밀도가 무너지는 곳
- **Stage 1 — 가장 중요한 결정은 criterion이 아니었음.** 식별 불가 파라미터(`theta`, `vth0`의 1/87 민감도)를 추정할지 공칭값에 고정할지였고, 공칭값 신뢰도에 따라 **32% 이득과 2배 손해** 사이를 오감. 손익분기는 공칭값 오차 20~30%이고, 손으로 쓴 두 규칙 모두 임계값을 다른 곳에 두어 **8건 중 3건씩 서로 반대 방향으로** 틀림
- **Stage 2 (코너)** — 공정 상관과 추정 상관은 같은 7×7 행렬이지만 다른 객체이고 `vth0`/`mu0`에서 **부호가 반대**(−0.33 대 +0.90, 21쌍 중 12쌍). 공정 공분산은 손으로 적지 않고 **물리적 원인 6개**(산화막 두께, 채널 길이·폭, 계면 트랩, 접촉 저항, 일함수)에서 유도
- **Stage 2 — 코너 2개가 128개를 이김.** 독립 박스 코너는 마진을 +50~81% 낭비하며 집단을 감싸지만, kσ 타원면을 겨냥하면 시뮬레이션 **2번**으로 거의 0% 낭비. 추출 노이즈 디컨볼루션은 **Stage 1이 식별 가능하다고 한 5개에서만** 작동(참 산포 10% 이내 복원)하고 `theta`·`vsat`는 9~10배 부풀려진 채 복원 불가 — 식별 불가 파라미터의 관측 산포는 공정 변동이 아니라 측정 노이즈
- **Stage 3 (회로) — 프로젝트 12의 소자 평가가 미수렴 전류를 반환하고 있었음.** 직렬 저항을 40회 상한 고정점 반복으로 풀어 nominal에서 38회(여유 2), `rs=2000`에서 500회 넘게 필요해 **72% 틀린 값**을 돌려줌. Newton(4회) + 음함수 정리 해석적 도함수로 교체하고, 프로젝트 12가 수렴한 구간에서 3e−14까지 일치함을 확인해 물리는 안 바뀌었음을 보장
- **Stage 3 — 모델에 음수 출력 컨덕턴스**가 동작 격자의 12%에 존재(rs=0에서는 14.5% — 직렬 저항이 결함을 부분적으로 가림). 프로젝트 12·13의 **어떤 단계도 이걸 잡을 수 없었음**(꺾이는 곡선도 자기 자신에는 잘 맞고, Fisher 해석은 `gds` 부호를 안 봄). 그런데 **수렴 실패를 한 건도 일으키지 않았고** 설계 문서는 일으킬 거라 예측 — `gds`가 지배하려면 필요한 부하가 소자를 음수 영역 밖으로 끌어내기 때문. Stage 2의 코너 결론은 실제 회로 측정에서도 유지(낭비 +54% 대 **+2%**)
- **Stage 3 — 결함 24건 중 3건은 trace로 판별 불가.** 솔브가 지나가지 않은 불연속은 건강한 회로와 종류상 같은 trace를 남기므로, 그 3건에 기권하면서 건강한 4건에는 기권하지 않는 진단자는 **존재할 수 없음**. 선택지는 **조용히 3번 틀리기**와 **7번 대답 거부**뿐이고, 어느 쪽이 나은지는 오진 비용에 대한 판단이지 trace에 대한 판단이 아님
- **AI는 두 가지 다른 모양의 판단에서 모두 규칙을 못 이김.** Stage 1(정량 임계값, 3조건 × 15회 반복)에서 한 번도 앞서지 못하고 매번 같은 자리·같은 방향으로 실패. Stage 3(이산 분류 + 기권 허용, 9회 반복)에서 규칙과 동점이고 **기권 0회** — 판별 불가 3건에서는 매 반복 만장일치로 오답. 다만 반복 간 분기가 **6개 클래스 중 2개에 집중**되고 9회 실행 두 번에서 재현됨(보정 후 p = 0.020, 0.0031). "긴 증거에서 흔들린다"는 설명은 배제(ρ ≈ +0.09)하고, 따라가는 건 그 판정이 몇 개 필드에 얽혀 있느냐(ρ = +0.66) — 예외가 있어 **메커니즘이 아니라 검증할 방향**으로 기록
- **등록한 예측 7개가 측정에 깨진 것을 표로 그대로 남김** — "AI가 식별 불가 파라미터를 빼서 이긴다"(빼는 쪽이 나빴음), "음수 `gds`가 Newton을 깨뜨린다"(두 조건이 레일 안에서 서로를 배제), "프록시 코너가 실제 지연을 under-cover한다"(같은 코너였음), "불리언 하나에 매달린 클래스가 불안정하다"(정반대였음) 등. 3회 반복 한 번으로 "설명 없는 증거가 해롭다"를 적었다가 15회에서 재현 안 돼 철회한 것도 포함
- 테스트 **235개**(Stage 1: 41, Stage 2: 23, Stage 3: 171). Jacobian은 자기가 미분이라 주장하는 잔차의 유한차분과, 소자 도함수는 실제 반환된 전류의 유한차분과, 집단 측정용 구적분 지연은 그걸 대신하는 Newton 과도 해석과 대조. 테스트를 쓰는 과정에서 **README의 틀린 주장 둘과 새 결과 하나**가 나옴
- 한계 명시: 전부 합성이고 정답을 알고 있음. EKV 형태 자체 구현으로 **BSIM 계열 아님**, 손으로 쓴 Newton은 **SPICE 아님**(source/gmin stepping 없음), 회로는 NMOS+부하만 — **음수 컨덕턴스가 문제됐을 전류미러 부하·캐스코드는 NMOS 전용 모델로 만들 수 없음**

→ [`projects/13_measurement-to-circuit`](./projects/13_measurement-to-circuit)


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
| 22 | Learning to Simulate Complex Physics with Graph Networks |
| 23 | A Comprehensive Review of Machine Learning Approaches for Semiconductor Device Modeling and Simulation |
| 24 | Neural Operators for Electromagnetics |
| 25 | Scientific Machine Learning for Semiconductor Manufacturing |
| 26 | Physics-Informed Machine Learning for Materials Science |
| 27 | DeePMD-kit: A Deep Learning Package for Many-Body Potential Energy Representation and Molecular Dynamics |
| 28 | End-to-end Symmetry Preserving Inter-atomic Potential Energy Model for Finite and Extended Systems (DeepPot-SE) |
| 29 | E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials (NequIP) |
| 30 | Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics (Allegro) |
| 31 | MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields |
| 32 | MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures |

---

## Next Steps

- [ ] PINN for 1D Heat / Poisson Equation (기초 PDE로 범위 확장)
- [ ] NEGF I-V 곡선 노이즈 개선 (격자 해상도/적응형 적분)
- [ ] MLIP force-matching 정확도 개선 (각도 항 descriptor 추가)
- [ ] AD-NEGF, DeePTB-NEGF 리뷰 후 ML 기반 NEGF 가속 실험
- [ ] Ion Implantation MC를 ZBL magic formula 기반 정밀 모델로 확장, channeling 효과 추가
- [ ] MeshGraphNet rollout 안정성 개선 (re-meshing, noise injection 등 Pfaff et al. 기법 적용) 및 2D 삼각형 mesh로 확장
- [ ] 프로젝트 13 Stage 1의 순차 재설계(sequential re-design) — 설계가 참값이 아니라 믿고 있는 값에서 만들어지는 간극을 메우는 루프
- [ ] 프로젝트 13 Stage 3 후속: 얽힌 결함 클래스에 답을 직접 주장하는 증거 필드를 주고 LLM의 분기가 **이동하는지** 측정 (움직이면 구조 가설, 안 움직이면 상관도 노이즈)
- [ ] Compact Model을 다중 geometry로 확장해 `theta`와 `rs`의 원리적 분리 불가 문제 해소 (프로젝트 12·13 공통 한계)

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
