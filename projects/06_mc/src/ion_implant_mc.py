"""
Ion Implantation Monte Carlo (simplified Binary Collision Approximation).

실제 반도체 공정(도핑)의 표준 단계인 이온주입을 대상으로 한다.
기본 예제: Boron(B) 이온을 Silicon(Si) 타겟에 주입.

모델링 방식 (BCA: Binary Collision Approximation):
1. 이온은 평균자유행로(mean free path) lambda 만큼 직선 비행 후 타겟 원자와 충돌
2. 충돌 시 전달 에너지 T는 power-law(Rutherford형) 미분단면적 dσ ∝ dT/T^2 근사에서
   inverse-CDF 샘플링으로 결정 (Sigmund의 power-potential 근사, LSS 이론 유도에 쓰이는
   표준 단순화 — 완전한 ZBL "magic formula"는 계수가 복잡해 여기서는 사용하지 않음)
3. 산란각은 2체 탄성충돌의 운동학적으로 정확한 관계식(T = gamma*E*sin^2(theta_c/2))으로
   부터 유도 — 이 부분은 근사가 아니라 엄밀한 운동학
4. 비행 구간마다 Lindhard-Scharff 전자적 정지능(S_e ∝ sqrt(E))으로 연속적 에너지 손실 적용
5. 에너지가 cutoff 이하로 떨어지거나 표면 밖으로 후방산란되면 궤적 종료, 최종 깊이 기록

명시적 단순화 (정직하게 문서화):
- 평균자유행로를 에너지에 무관한 상수(원자 간격 스케일)로 고정 — 실제로는 E에 따라
  달라지는 총 단면적에서 유도되어야 함
- 결정 구조(channeling 효과) 무시, 비정질(amorphous) 타겟 가정
- 완전한 ZBL 스크리닝 포텐셜 대신 power-law 미분단면적 근사 사용

구현 노트: 성능을 위해 방향 벡터를 numpy 배열 대신 순수 파이썬 float 튜플로 다룬다
(짧은 배열에 대한 numpy 함수 호출 오버헤드가 반복 횟수가 많을 때 병목이 되기 때문).
"""

import math
import numpy as np

# --- 물리 상수 ---
E_COULOMB = 14.4  # eV*Angstrom

# --- 타겟: Silicon ---
Z2 = 14
M2 = 28.09  # amu
N_SI_CM3 = 5.0e22  # atoms/cm^3
N_SI_ANG3 = N_SI_CM3 * 1e-24  # atoms/Angstrom^3
MEAN_SPACING = N_SI_ANG3 ** (-1.0 / 3.0)  # Angstrom

# --- 이온: Boron ---
Z1 = 5
M1 = 10.81  # amu

GAMMA = 4 * M1 * M2 / (M1 + M2) ** 2
MASS_RATIO = M1 / M2

E_CUTOFF = 5.0
T_MIN = 2.0
K_ELECTRONIC = 0.15


def sample_energy_transfer(E, u):
    """dσ ∝ dT/T^2 미분단면적에서 inverse-CDF 샘플링. u는 [0,1) 균일난수."""
    T_max = GAMMA * E
    if T_max <= T_MIN:
        return T_max
    inv_T = 1.0 / T_MIN - u * (1.0 / T_MIN - 1.0 / T_max)
    T = 1.0 / inv_T
    return T if T < T_max else T_max


def simulate_ion(E0, rng, max_collisions=4000, record_path=False):
    """
    단일 이온의 궤적을 시뮬레이션 (순수 파이썬 float 연산, 성능 최적화).
    반환: final_depth (z, Angstrom), backscattered (bool), n_collisions, path
    """
    E = E0
    x, y, z = 0.0, 0.0, 0.0
    dx, dy, dz = 0.0, 0.0, 1.0  # z축 방향으로 입사

    path = [(x, y, z)] if record_path else None

    randoms = rng.random(size=(max_collisions, 3))  # [electronic 관련 없음, T용, phi용] 사전 생성

    for i in range(max_collisions):
        S_e = K_ELECTRONIC * math.sqrt(E) if E > 0 else 0.0
        E -= S_e * MEAN_SPACING
        if E <= E_CUTOFF:
            break

        x += MEAN_SPACING * dx
        y += MEAN_SPACING * dy
        z += MEAN_SPACING * dz
        if record_path:
            path.append((x, y, z))

        if z < 0:
            return z, True, i + 1, path

        T = sample_energy_transfer(E, randoms[i, 1])
        E -= T
        if E <= E_CUTOFF:
            break

        ratio = T / (GAMMA * (E + T))
        ratio = 0.0 if ratio < 0 else (1.0 if ratio > 1 else ratio)
        theta_c = 2 * math.asin(math.sqrt(ratio))
        sin_c, cos_c = math.sin(theta_c), math.cos(theta_c)
        denom = MASS_RATIO + cos_c
        theta_lab = math.atan2(sin_c, denom) if denom != 0 else math.pi / 2
        phi = 2 * math.pi * randoms[i, 2]

        # 현재 방향 (dx,dy,dz)를 theta_lab, phi만큼 회전 (로컬 좌표계 구성)
        if abs(dz) < 0.999:
            rx, ry, rz = 0.0, 0.0, 1.0
        else:
            rx, ry, rz = 1.0, 0.0, 0.0
        # u1 = d x ref
        u1x = dy * rz - dz * ry
        u1y = dz * rx - dx * rz
        u1z = dx * ry - dy * rx
        norm1 = math.sqrt(u1x**2 + u1y**2 + u1z**2)
        u1x, u1y, u1z = u1x / norm1, u1y / norm1, u1z / norm1
        # u2 = d x u1
        u2x = dy * u1z - dz * u1y
        u2y = dz * u1x - dx * u1z
        u2z = dx * u1y - dy * u1x

        cos_t, sin_t = math.cos(theta_lab), math.sin(theta_lab)
        cos_p, sin_p = math.cos(phi), math.sin(phi)
        ndx = cos_t * dx + sin_t * (cos_p * u1x + sin_p * u2x)
        ndy = cos_t * dy + sin_t * (cos_p * u1y + sin_p * u2y)
        ndz = cos_t * dz + sin_t * (cos_p * u1z + sin_p * u2z)
        norm = math.sqrt(ndx**2 + ndy**2 + ndz**2)
        dx, dy, dz = ndx / norm, ndy / norm, ndz / norm

    return z, False, i + 1, path


def run_ensemble(E0, n_ions, seed=0, max_collisions=4000):
    rng = np.random.default_rng(seed)
    depths = np.empty(n_ions)
    n_valid = 0
    backscatter_count = 0
    for k in range(n_ions):
        depth, backscattered, n_coll, _ = simulate_ion(E0, rng, max_collisions=max_collisions)
        if backscattered:
            backscatter_count += 1
        else:
            depths[n_valid] = depth
            n_valid += 1
    return depths[:n_valid], backscatter_count


if __name__ == "__main__":
    import time
    print(f"Target (Si): atomic density = {N_SI_ANG3:.4f} /Ang^3, mean spacing = {MEAN_SPACING:.3f} Ang")
    print(f"Ion (B->Si): gamma (max energy transfer fraction) = {GAMMA:.4f}\n")

    for E0_keV in [30, 60, 100]:
        E0 = E0_keV * 1000
        t0 = time.time()
        depths, n_back = run_ensemble(E0, n_ions=500, seed=42)
        elapsed = time.time() - t0
        Rp = depths.mean() / 10
        dRp = depths.std() / 10
        print(f"E0={E0_keV} keV: Rp={Rp:.1f} nm, dRp={dRp:.1f} nm (dRp/Rp={dRp/Rp:.2f}), "
              f"backscattered={n_back}/500  [{elapsed:.1f}s]")
