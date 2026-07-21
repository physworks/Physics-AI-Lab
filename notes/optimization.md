# Optimization Notes

## Physics Loss + Data Loss (PINN)

데이터가 희소한 영역에서 물리 방정식을 collocation point로 활용해 제약을 추가. 순수 data loss만 쓰는 baseline 대비, 데이터가 부족한 영역에서 일반화 성능이 개선된다.

- 주의: 물리 제약이 항상 유리한 것은 아니다. GCA-PINN에서 물리 방정식 자체의 불연속을 제거하자(더 매끄러워지자) baseline과의 격차가 14.9%→5.2%로 줄었다 — **물리 제약은 데이터만으로 배우기 어려운(불연속, sharp feature) 영역에서 가장 유용하다.**
- 구현: `projects/01_gca-pinn/src/train.py`

## Energy Loss + Force Loss (MLIP)

에너지만 맞추도록 학습하면 힘의 방향(에너지의 미분 구조)까지 정확하다는 보장이 없다. Force loss를 추가로 걸어야 하며, 초기 실험에서 force loss 가중치가 너무 작으면(에너지 손실 규모에 압도되어) force loss가 전혀 줄지 않는 현상을 확인했다 — loss 항목 간 스케일 균형이 중요.

- 구현: `projects/04_mlip/src/train_mlip.py`

## Gummel Iteration (Fixed-point vs Newton-linearized)

Poisson과 Drift-Diffusion을 번갈아 푸는 decoupled iteration에서, Poisson step을 순수 fixed-point(전하 고정 선형 풀이)로 하면 발산한다. 준-페르미 준위를 고정한 채 비선형 Poisson을 Newton으로 푸는 것이 안정성의 핵심 — 전하의 전위 민감도(exponential feedback)를 Jacobian에 반영해야 수렴한다.

## Newton-Raphson Damping

강한 비선형 시스템(지수함수적 전하밀도)에서 step 크기를 제한하지 않으면 발산하기 쉽다. `dphi`의 최대 크기가 임계값을 넘으면 정규화하는 간단한 damping으로 안정적인 수렴을 확보.
