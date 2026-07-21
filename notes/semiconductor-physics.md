# Semiconductor Physics Notes

## TFT / GCA
- Gradual Channel Approximation derivation
- Linear vs saturation region boundary condition
- Sources of deviation from ideal GCA in real TFTs (subthreshold, mobility degradation, interface traps)

## MOS Capacitor (Poisson)
- 평형 상태 비선형 Poisson: d/dx(eps dphi/dx) = -q[p(phi)-n(phi)-Na], Boltzmann 통계 가정
- Quasi-static C-V 곡선의 3단계: 축적(accumulation, 고용량) → 공핍(depletion, 용량 감소) → 반전(inversion, 용량 회복)
- 문턱전압 근처(2φ_F)에서 C-V 최솟값

## PN Junction Diode (Drift-Diffusion)
- 전자/정공 연속방정식: dJn/dx = qR, dJp/dx = -qR (이상적 다이오드는 R=0 가정)
- Shockley 지수 법칙: 순방향 전류가 exp(Va/Vt)로 증가 (18자리에 걸쳐 검증)
- 경계조건: 다수캐리어=도핑농도, 소수캐리어=평형값×exp(Va/Vt) ("law of the junction")

## Resonant Tunneling Diode (NEGF)
- Double-barrier 구조의 양자 우물이 만드는 준속박상태가 투과율 T(E)에 공명 피크로 나타남
- 유한한 장벽 높이에서는 리드와의 결합으로 준속박상태가 broadening/이동 (고립계 근사와의 차이)
- NDR(음의 미분저항): 바이어스에 따라 공명 준위가 이동해 전도 창(conduction window)과의 정렬이 깨지며 발생. 대칭 구조에서는 이 효과가 상쇄되어 사라질 수 있음 — 소자 형상의 비대칭성이 실제로 중요.

## Lennard-Jones MD / MLIP
- LJ potential: U(r) = 4ε[(σ/r)¹²-(σ/r)⁶], reduced units(ε=σ=m=1) 표준 관행
- 원자 국소 환경을 순서·회전 불변 descriptor로 표현 (Behler-Parrinello symmetry functions)
- Force-matching이 energy-matching보다 근본적으로 어려움 — 에너지는 스칼라 하나, 힘은 에너지 곡면의 미분(기울기) 정확도까지 요구

