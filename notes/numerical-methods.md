# Numerical Methods Notes

## Box-Integration (FVM) ≡ 1D Linear FEM

각 노드 주변에 제어체적(control volume)을 잡고 flux balance를 세우는 방식. 1D 균일 계수 문제에서는 표준 선형 FEM(Galerkin)과 수학적으로 동일한 강성행렬을 만든다. Sentaurus를 포함한 대부분의 상용 TCAD가 실제로 채택하는 이산화 기법.

- 구현: `projects/03_tcad-surrogate/src/poisson_1d.py`
- 핵심: 계면 근처는 Debye length 스케일로 촘촘하게, 벌크로 갈수록 기하급수적으로 성기게 하는 비균일 메시가 정확도에 중요.

## Newton-Raphson (비선형 Poisson)

전하밀도가 전위에 대해 지수함수이므로 강한 비선형 시스템. 매 반복마다 Jacobian(dcharge/dphi = -(q/Vt)(n+p))을 직접 조립하고 damped step으로 수렴.

## Scharfetter-Gummel Discretization

전자/정공 전류밀도를 노드 간 준-페르미 준위가 지수함수적으로 변한다고 가정해 이산화하는 표준 기법. Bernoulli function B(x)=x/(e^x-1)을 이용.

- 구현: `projects/03_tcad-surrogate/src/diode_dd.py`

## Gummel Iteration

Poisson과 Drift-Diffusion을 번갈아 self-consistent하게 수렴시키는 decoupled 반복법 (Gummel, 1964). Poisson step에서 순수하게 전하를 고정한 선형 풀이를 하면 발산한다 — 이전 스텝의 준-페르미 준위를 고정한 채 비선형 Poisson을 Newton으로 푸는 것이 핵심(전하의 전위 민감도를 Jacobian에 반영해야 안정적).

## 행렬 조건수(Ill-conditioning)와 Shooting Method

전자 농도가 경계(소수캐리어)부터 벌크(다수캐리어)까지 19자리 가까이 차이 나면, 일반 dense 행렬 풀이가 double precision(약 15~16자리)의 한계를 넘어 발산할 수 있다. "전류가 상수"라는 물리적 사실(dJ/dx=0)을 이용해 N차원 선형계 대신 전류 값 하나(스칼라)에 대한 shooting method(scipy.optimize.brentq)로 문제를 재구성하면 이 문제를 원천적으로 피할 수 있다.

## Symmetry Function Descriptor (MLIP)

원자의 국소 환경을 순서·회전 불변인 고정 길이 벡터로 표현하는 방법 (Behler-Parrinello, 2007). Gaussian 기준점들에서 이웃 원자까지의 거리를 평가해 합산. torch로 구현 시 self-distance를 inf로 마스킹하면 backward pass에서 NaN이 발생하는 함정에 주의 (masked_fill 대신 명시적 마스크 곱셈 사용).

- 구현: `projects/04_mlip/src/descriptors.py`

## NEGF: Self-Energy와 Surface Green's Function

무한한 리드를 self-energy로 "요약"해 유한 크기 행렬 문제로 축소하는 것이 NEGF의 핵심. 반무한 1D tight-binding 체인은 표면 Green's function이 닫힌 형태로 알려져 있어(g(E) = (E-i√(4t²-E²))/(2t²)), 반복 계산(Sancho-Rubio) 없이 exact하게 self-energy를 구할 수 있다.

- 구현: `projects/05_negf/src/negf_core.py`
