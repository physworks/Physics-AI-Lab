"""
Bayesian Optimization: from-scratch Gaussian Process 회귀 + Expected Improvement 획득함수.

MLIP/MeshGraphNet 프로젝트에서 신경망을 직접 구현한 것과 같은 방식으로,
scikit-learn 등에 의존하지 않고 GP 회귀와 EI 획득함수를 직접 구현한다.

GP 회귀: RBF(squared-exponential) 커널, closed-form posterior mean/variance.
Expected Improvement: 현재까지의 최적값(best-so-far) 대비 개선 기댓값이
가장 큰 지점을 다음 평가 후보로 선택 — exploration(불확실성이 큰 곳)과
exploitation(예측이 좋은 곳)의 균형을 자동으로 맞추는 표준 acquisition function.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def rbf_kernel(X1, X2, length_scale, sigma_f):
    """X1: (n1,d), X2: (n2,d) -> (n1,n2) 커널 행렬"""
    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * X1 @ X2.T
    return sigma_f ** 2 * np.exp(-0.5 * sqdist / length_scale ** 2)


class GaussianProcess:
    def __init__(self, length_scale=0.3, sigma_f=1.0, noise=1e-6):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.noise = noise

    def fit(self, X, y):
        self.X_train = X
        self.y_mean = y.mean()
        self.y_train = y - self.y_mean
        K = rbf_kernel(X, X, self.length_scale, self.sigma_f) + self.noise * np.eye(len(X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))

    def predict(self, X_star):
        K_star = rbf_kernel(X_star, self.X_train, self.length_scale, self.sigma_f)
        mu = K_star @ self.alpha + self.y_mean
        v = np.linalg.solve(self.L, K_star.T)
        K_star_star = rbf_kernel(X_star, X_star, self.length_scale, self.sigma_f)
        cov = K_star_star - v.T @ v
        var = np.clip(np.diag(cov), 1e-12, None)
        return mu, np.sqrt(var)


def expected_improvement(X_cand, gp, y_best, xi=0.01):
    """y_best: 지금까지 관측된 최솟값 (minimize 문제이므로 개선 = y_best - mu)"""
    mu, sigma = gp.predict(X_cand)
    improvement = y_best - mu - xi
    Z = improvement / sigma
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


def propose_next_point(gp, y_best, bounds, n_restarts=10, rng=None):
    """EI를 최대화하는 다음 평가 지점을 다중 시작점 국소 최적화로 탐색."""
    rng = rng or np.random.default_rng()
    dim = len(bounds)
    best_x, best_val = None, -np.inf

    def neg_ei(x):
        return -expected_improvement(x.reshape(1, -1), gp, y_best)[0]

    for _ in range(n_restarts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        res = minimize(neg_ei, x0, bounds=bounds, method="L-BFGS-B")
        if -res.fun > best_val:
            best_val = -res.fun
            best_x = res.x
    return best_x


def bayesian_optimize(objective, bounds, n_init=5, n_iter=20, seed=0):
    """
    objective: callable, x(array of dim D) -> scalar loss (minimize)
    bounds: [(lo,hi), ...] 파라미터별 범위
    반환: X_history, y_history (평가한 모든 점과 값)
    """
    rng = np.random.default_rng(seed)
    dim = len(bounds)

    X = np.array([[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(n_init)])
    y = np.array([objective(x) for x in X])

    gp = GaussianProcess(length_scale=0.3, sigma_f=max(y.std(), 1e-3))

    for it in range(n_iter):
        # 정규화된 입력으로 GP 학습 (파라미터 스케일이 다를 수 있으므로)
        X_norm = normalize(X, bounds)
        gp.fit(X_norm, y)
        y_best = y.min()

        norm_bounds = [(0.0, 1.0)] * dim
        x_next_norm = propose_next_point(gp, y_best, norm_bounds, rng=rng)
        x_next = denormalize(x_next_norm, bounds)

        y_next = objective(x_next)
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

    return X, y


def normalize(X, bounds):
    X = np.atleast_2d(X)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return (X - lo) / (hi - lo)


def denormalize(x_norm, bounds):
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return lo + x_norm * (hi - lo)


if __name__ == "__main__":
    # 간단한 2D 테스트 함수로 GP+BO 자체를 검증 (TCAD와 무관, 순수 알고리즘 검증)
    def test_func(x):
        return (x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2

    bounds = [(0.0, 1.0), (0.0, 1.0)]
    X, y = bayesian_optimize(test_func, bounds, n_init=5, n_iter=15, seed=1)
    best_idx = np.argmin(y)
    print(f"True minimum at (0.3, 0.7), value=0")
    print(f"BO found: x={X[best_idx]}, y={y[best_idx]:.6f} after {len(y)} evaluations")
