"""
Kozachenko–Leonenko k-NN differential entropy and Eq. (6)-style AU/EU for Gaussian mixtures.

References:
    Kozachenko, L. F., & Leonenko, N. N. (1987). Sample estimate of the entropy of a random vector.
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information.

Finite sample sizes induce bias and variance; epistemic estimates can be negative.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma
from scipy.spatial import cKDTree


def kozachenko_leonenko_entropy_1d(samples: np.ndarray, k: int = 3) -> float:
    """
    Kozachenko–Leonenko estimate of differential entropy (nats) for univariate data.

    Uses Euclidean distance in R^1, k-th nearest neighbor excluding the query point.

    Parameters
    ----------
    samples : array-like, shape (L,)
        i.i.d. draws from a continuous distribution.
    k : int
        Neighbor order (must satisfy L > k >= 1).

    Returns
    -------
    float
        Estimated differential entropy in nats, or np.nan if L <= k.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    n = x.size
    if n <= k or k < 1:
        return float(np.nan)

    X = x.reshape(-1, 1)
    d = 1
    tree = cKDTree(X)
    dist, _ = tree.query(X, k=k + 1)
    rho = np.maximum(dist[:, k], np.finfo(np.float64).tiny)

    log_Vd = np.log(2.0)  # volume of L2 unit ball in R^1
    return float(digamma(n) - digamma(k) + log_Vd + (d / n) * np.sum(np.log(rho)))


def entropy_uncertainty_knn_gaussian_mixture(
    mu: np.ndarray,
    sigma2: np.ndarray,
    L: int = 5000,
    k_nn: int = 3,
    rng: np.random.Generator | None = None,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    """
    Eq. (6)-style decomposition using k-NN entropy estimates.

    For each grid point x_n, members i = 1..M give Gaussians N(mu_{i,n}, sigma2_{i,n}).
    The marginal predictive is an equal-weight mixture over members.

    - Draw y^(1),...,y^(L) ~ p(y|x_n) (sample member uniformly, then Gaussian).
    - H_tot,n = KL estimate from those L samples.
    - For each member i, draw L samples from N(mu_{i,n}, sigma2_{i,n}), H_i,n = KL estimate.
    - AU_n = (1/M) sum_i H_i,n, EU_n = H_tot,n - AU_n, TU_n = H_tot,n.

    Parameters
    ----------
    mu : array-like, shape (M, N)
        Per-member predictive means.
    sigma2 : array-like, shape (M, N)
        Per-member predictive variances.
    L : int
        Monte Carlo sample size per entropy estimate (paper's L).
    k_nn : int
        k for Kozachenko–Leonenko (must satisfy L > k_nn).
    rng : np.random.Generator, optional
    eps : float
        Floor on sigma2 when sampling.

    Returns
    -------
    dict with keys 'aleatoric', 'epistemic', 'total', each shape (N,).
    """
    rng = rng or np.random.default_rng()
    mu = np.asarray(mu, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    if mu.ndim == 1:
        mu = mu.reshape(1, -1)
    if sigma2.ndim == 1:
        sigma2 = sigma2.reshape(1, -1)
    if L <= k_nn:
        raise ValueError(f"L ({L}) must be greater than k_nn ({k_nn}).")

    M, N = mu.shape
    if sigma2.shape != (M, N):
        raise ValueError(f"sigma2 shape {sigma2.shape} does not match mu {(M, N)}")

    sigma2_safe = np.maximum(sigma2, eps)

    idx = rng.integers(0, M, size=(L, N), endpoint=False)
    col = np.arange(N, dtype=np.int64)
    mu_s = mu[idx, col]
    sig_s = sigma2_safe[idx, col]
    z = rng.standard_normal(size=(L, N))
    y_mix = z * np.sqrt(sig_s) + mu_s

    total = np.empty(N, dtype=np.float64)
    for n in range(N):
        total[n] = kozachenko_leonenko_entropy_1d(y_mix[:, n], k_nn)

    H_members = np.empty((M, N), dtype=np.float64)
    for i in range(M):
        std = np.sqrt(sigma2_safe[i, :])
        y_i = rng.standard_normal(size=(L, N)) * std + mu[i, :]
        for n in range(N):
            H_members[i, n] = kozachenko_leonenko_entropy_1d(y_i[:, n], k_nn)

    aleatoric = H_members.mean(axis=0)
    epistemic = total - aleatoric

    return {
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total": total,
    }
