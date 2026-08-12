"""
Scoring the overall predictive distribution of an equally-weighted Gaussian mixture

    p(y|x) = (1/M) * sum_m N(y; mu_m(x), sigma_m(x)^2)

built from a method's per-member/per-draw predictions (mu_m, sigma_m), independent of
the aleatoric/epistemic/total variance decomposition used elsewhere in this repo.

Provides: the 4 canonical toy-regression DGPs (true mean/sigma, matching
Experiments/*.ipynb's generate_toy_regression exactly -- NOT utils.metrics.
compute_true_noise_variance, which is wrong for the homoscedastic case), mixture
CDF/log-density/quantiles, closed-form mixture CRPS (Grimit et al. 2006), mixture NLL,
PIT, coverage (marginal + stratified), sharpness, oracle (true-DGP) reference metrics,
and an ensemble-member subsampling helper for the Deep Ensemble size sweep.

Pure numpy/scipy -- no torch or rpy2 import, so this module stays fast to import and
trivially unit-testable in isolation.
"""

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf, logsumexp


# ============================== DGP table ==============================

@dataclass(frozen=True)
class DGPSpec:
    func_type: str
    noise_type: str
    mean_fn: Callable[[np.ndarray], np.ndarray]
    sigma_fn: Callable[[np.ndarray], np.ndarray]


def _linear_mean(x):
    return 0.7 * np.asarray(x, dtype=float) + 0.5


def _sin_mean(x):
    x = np.asarray(x, dtype=float)
    return x * np.sin(x) + x


def _homoscedastic_sigma(x):
    return np.full_like(np.asarray(x, dtype=float), 1.0)


def _heteroscedastic_sigma(x):
    x = np.asarray(x, dtype=float)
    return np.abs(2.5 * np.sin(0.5 * x + 5))


_MEAN_FNS = {'linear': _linear_mean, 'sin': _sin_mean}
_SIGMA_FNS = {'homoscedastic': _homoscedastic_sigma, 'heteroscedastic': _heteroscedastic_sigma}

DGP_TABLE: Dict[Tuple[str, str], DGPSpec] = {
    (func_type, noise_type): DGPSpec(func_type, noise_type, _MEAN_FNS[func_type], _SIGMA_FNS[noise_type])
    for func_type in ('linear', 'sin')
    for noise_type in ('homoscedastic', 'heteroscedastic')
}


def get_dgp(func_type: str, noise_type: str) -> DGPSpec:
    key = (func_type, noise_type)
    if key not in DGP_TABLE:
        raise ValueError(f"Unknown DGP combination: {key}. Must be one of {sorted(DGP_TABLE)}")
    return DGP_TABLE[key]


def dgp_name(func_type: str, noise_type: str) -> str:
    return f"{func_type}-{noise_type}"


# ============================== Data generation ==============================

def generate_training_data(func_type: str, noise_type: str, n_train: int = 1000,
                            train_range: Tuple[float, float] = (-5, 10), seed: int = 42
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduces Experiments/*.ipynb's generate_toy_regression exactly (consumes the
    global np.random stream, matching notebook convention). Returns
    (x_train, y_train, y_clean_train), all float32 shape (n_train, 1)."""
    spec = get_dgp(func_type, noise_type)
    np.random.seed(seed)
    low, high = train_range
    x_train = np.random.uniform(low, high, size=(n_train, 1))
    y_clean = spec.mean_fn(x_train)
    sigma_train = spec.sigma_fn(x_train)
    epsilon = np.random.normal(0.0, sigma_train, size=(n_train, 1))
    y_train = y_clean + epsilon
    return x_train.astype(np.float32), y_train.astype(np.float32), y_clean.astype(np.float32)


def make_test_set(func_type: str, noise_type: str, x_range: Tuple[float, float] = (-5, 10),
                   n_test: int = 300, test_seed: int = 20260809
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic dense grid of held-out test x's, with a noisy y_test drawn from an
    INDEPENDENT np.random.default_rng(test_seed) stream -- deliberately not the global
    np.random state, so the test set is identical regardless of what training/model code
    ran before it, even across methods/DGPs in a batch sweep.

    Returns (x_test, y_test, mu_true, sigma_true), all float32 shape (n_test, 1)."""
    spec = get_dgp(func_type, noise_type)
    x_test = np.linspace(x_range[0], x_range[1], n_test).reshape(-1, 1).astype(np.float32)
    mu_true = spec.mean_fn(x_test).astype(np.float32)
    sigma_true = spec.sigma_fn(x_test).astype(np.float32)
    rng = np.random.default_rng(test_seed)
    y_test = (mu_true + rng.normal(0.0, sigma_true, size=x_test.shape)).astype(np.float32)
    return x_test, y_test, mu_true, sigma_true


# ============================== Shape normalization ==============================

def normalize_mixture_arrays(mu_samples: np.ndarray, sigma2_samples: np.ndarray,
                              n_expected: int) -> Tuple[np.ndarray, np.ndarray]:
    """Orient a (mu_samples, sigma2_samples) pair to shape (M, n_expected). Defensive
    insurance against per-method axis-order drift (e.g. BAMLSS historically returning
    (N, S) in some code paths) -- disambiguates using the known test-set size rather
    than trusting a hardcoded per-model convention."""
    mu = np.asarray(mu_samples)
    sigma2 = np.asarray(sigma2_samples)
    if mu.ndim != 2 or sigma2.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got shapes {mu.shape}, {sigma2.shape}")
    if mu.shape != sigma2.shape:
        raise ValueError(f"mu_samples shape {mu.shape} != sigma2_samples shape {sigma2.shape}")
    if mu.shape[1] == n_expected:
        return mu, sigma2
    if mu.shape[0] == n_expected:
        return mu.T, sigma2.T
    raise ValueError(f"Cannot orient shape {mu.shape} to have an axis of size n_expected={n_expected}")


# ============================== Mixture primitives ==============================

def mixture_cdf(y, mu, sigma2):
    """F(y) = mean_m Phi((y - mu_m) / sigma_m). Broadcasts y against the leading (member)
    axis of mu/sigma2: mu/sigma2 shape (M,) with scalar y (single-x, used inside brentq),
    or shape (M, N) with y shape (N,) (vectorized PIT/CRPS case)."""
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.sqrt(np.maximum(sigma2, 1e-300))
    z = (y[np.newaxis, ...] - mu) / sigma
    return np.mean(0.5 * (1.0 + erf(z / np.sqrt(2.0))), axis=0)


def mixture_log_pdf(y, mu, sigma2):
    """log p(y|x) via log-sum-exp over the member axis. y: (N,), mu/sigma2: (M, N)."""
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma2 = np.maximum(sigma2, 1e-300)
    M = mu.shape[0]
    log_component = -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * (y[np.newaxis, ...] - mu) ** 2 / sigma2
    return logsumexp(log_component, axis=0) - np.log(M)


def mixture_nll(y, mu, sigma2) -> float:
    return float(-np.mean(mixture_log_pdf(y, mu, sigma2)))


def compute_pit(y, mu, sigma2) -> np.ndarray:
    """PIT values F_hat(y_i|x_i) using the exact mixture CDF."""
    return mixture_cdf(y, mu, sigma2)


def mixture_quantile(p: float, mu, sigma2, bracket_pad: float = 4.0, xtol: float = 1e-8,
                      rtol: float = 1e-10, maxiter: int = 200) -> float:
    """Root of mixture_cdf(y, mu, sigma2) - p == 0 for ONE x's components (1D mu/sigma2,
    length M). Bracket = [min(mu) - bracket_pad*max(sigma), max(mu) + bracket_pad*max(sigma)],
    valid since the mixture CDF is monotone non-decreasing."""
    mu = np.asarray(mu, dtype=float).ravel()
    sigma2 = np.asarray(sigma2, dtype=float).ravel()
    sigma = np.sqrt(np.maximum(sigma2, 1e-300))
    lo = mu.min() - bracket_pad * sigma.max()
    hi = mu.max() + bracket_pad * sigma.max()

    def objective(y):
        return float(mixture_cdf(y, mu, sigma2)) - p

    return brentq(objective, lo, hi, xtol=xtol, rtol=rtol, maxiter=maxiter)


def mixture_quantiles_for_levels(mu, sigma2, levels: Iterable[float], **brentq_kwargs
                                  ) -> Dict[float, np.ndarray]:
    """Vectorized over test points (mu, sigma2 shape (M, N)): one brentq root-find per
    (level, test point) -- M is small (<=30) so this is cheap. Returns {level: (N,) array}."""
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    _, N = mu.shape
    levels = sorted(set(levels))
    out = {lv: np.empty(N) for lv in levels}
    for j in range(N):
        mu_j = mu[:, j]
        sigma2_j = sigma2[:, j]
        for lv in levels:
            out[lv][j] = mixture_quantile(lv, mu_j, sigma2_j, **brentq_kwargs)
    return out


# ============================== Closed-form mixture CRPS (Grimit et al. 2006) ==============

def _abs_moment_gaussian(mu, sigma):
    """A(mu, sigma) = E|N(mu, sigma^2)| = mu*(2*Phi(mu/sigma) - 1) + 2*sigma*phi(mu/sigma)
    for sigma > 0, and |mu| in the sigma -> 0 limit. Symmetric under mu -> -mu."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_safe = np.where(sigma > 1e-12, sigma, 1.0)
    z = mu / sigma_safe
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    formula = mu * (2 * cdf - 1) + 2 * sigma_safe * pdf
    return np.where(sigma > 1e-12, formula, np.abs(mu))


def mixture_crps(y, mu, sigma2) -> np.ndarray:
    """Per-test-point closed-form CRPS of an equally-weighted Gaussian mixture, via the
    energy-distance identity CRPS(F,y) = E_F|X-y| - (1/2)E_F|X-X'|:

        CRPS(F,y) = (1/M) sum_i A(y-mu_i, sigma_i)
                    - (1/(2M^2)) sum_i sum_j A(mu_i-mu_j, sqrt(sigma_i^2+sigma_j^2))

    y: (N,), mu/sigma2: (M, N). Returns (N,) array; caller takes .mean() for the scalar
    CRPS. O(M^2 * N), fine at M<=30, N~hundreds."""
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.sqrt(np.maximum(sigma2, 0.0))

    term1 = _abs_moment_gaussian(y[np.newaxis, :] - mu, sigma).mean(axis=0)

    pair_mu = mu[:, np.newaxis, :] - mu[np.newaxis, :, :]
    pair_sigma = np.sqrt(sigma2[:, np.newaxis, :] + sigma2[np.newaxis, :, :])
    term2 = 0.5 * _abs_moment_gaussian(pair_mu, pair_sigma).mean(axis=(0, 1))

    return term1 - term2


# ============================== Coverage / sharpness ==============================

def empirical_coverage(y, quantile_dict: Dict[float, np.ndarray],
                        nominal_levels: Iterable[float] = (0.5, 0.8, 0.9, 0.95)) -> Dict[float, float]:
    y = np.asarray(y, dtype=float).ravel()
    coverage = {}
    for a in nominal_levels:
        lo = quantile_dict[(1 - a) / 2]
        hi = quantile_dict[(1 + a) / 2]
        coverage[a] = float(np.mean((y >= lo) & (y <= hi)))
    return coverage


def sharpness_stats(quantile_dict: Dict[float, np.ndarray], sigma2_star) -> Dict[str, float]:
    width_90 = quantile_dict[0.95] - quantile_dict[0.05]
    return {
        'mean_width_90': float(np.mean(width_90)),
        'median_width_90': float(np.median(width_90)),
        'mean_pred_std': float(np.mean(np.sqrt(np.maximum(np.asarray(sigma2_star, dtype=float), 0.0)))),
    }


def stratified_coverage(x, y, quantile_dict: Dict[float, np.ndarray],
                         nominal_levels: Iterable[float] = (0.5, 0.8, 0.9, 0.95),
                         n_bins: int = 10) -> List[dict]:
    """Empirical coverage within quantile-based (equal-frequency) bins of x. Marginal
    calibration can mask region-specific miscalibration."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    n_actual_bins = len(edges) - 1
    if n_actual_bins < 1:
        return []
    bin_idx = np.clip(np.digitize(x, edges[1:-1], right=True), 0, n_actual_bins - 1)

    rows = []
    for b in range(n_actual_bins):
        mask = bin_idx == b
        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue
        for a in nominal_levels:
            lo = quantile_dict[(1 - a) / 2][mask]
            hi = quantile_dict[(1 + a) / 2][mask]
            emp_cov = float(np.mean((y[mask] >= lo) & (y[mask] <= hi)))
            rows.append({
                'bin_index': b, 'bin_low': float(edges[b]), 'bin_high': float(edges[b + 1]),
                'n_in_bin': n_in_bin, 'nominal_level': a, 'empirical_coverage': emp_cov,
            })
    return rows


# ============================== Oracle (true DGP, single Gaussian) ==============================

def oracle_metrics(y_test, mu_true, sigma_true) -> Dict[str, float]:
    """CRPS/NLL under the TRUE DGP conditional distribution N(mu_true, sigma_true^2) --
    an achievable floor, so mixture scores are interpretable in absolute terms. The
    oracle is genuinely a single Gaussian, so this is the one place it's correct to
    reuse utils.metrics' single-Gaussian closed forms."""
    from utils.metrics import compute_crps_gaussian, compute_gaussian_nll
    sigma2_true = np.asarray(sigma_true, dtype=float) ** 2
    return {
        'oracle_crps': float(compute_crps_gaussian(y_test, mu_true, sigma2_true)),
        'oracle_nll': float(compute_gaussian_nll(y_test, mu_true, sigma2_true)),
    }


# ============================== One-call evaluator ==============================

def evaluate_mixture(x_test, y_test, mu, sigma2, coverage_levels: Iterable[float] = (0.5, 0.8, 0.9, 0.95),
                      n_bins: int = 10, include_stratified: bool = True, **brentq_kwargs) -> dict:
    """Computes RMSE, CRPS, NLL, PIT, coverage (marginal + optionally stratified), and
    sharpness for one (method, DGP, scenario, replicate) cell.

    Returns {'scalar': {rmse, crps, nll, coverage_<level>..., mean_width_90,
    median_width_90, mean_pred_std}, 'pit': (N,) array, 'stratified': list[dict]}."""
    from utils.metrics import compute_predictive_aggregation

    x_test = np.asarray(x_test, dtype=float).ravel()
    y_test = np.asarray(y_test, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    coverage_levels = list(coverage_levels)

    mu_star, sigma2_star = compute_predictive_aggregation(mu, sigma2)
    rmse = float(np.sqrt(np.mean((y_test - mu_star) ** 2)))
    crps = float(mixture_crps(y_test, mu, sigma2).mean())
    nll = mixture_nll(y_test, mu, sigma2)
    pit = compute_pit(y_test, mu, sigma2)

    # Always include 0.05/0.95 so the fixed central-90% sharpness interval is available
    # regardless of the requested coverage_levels.
    levels_needed = sorted({lv for a in coverage_levels for lv in ((1 - a) / 2, (1 + a) / 2)} | {0.05, 0.95})
    quantile_dict = mixture_quantiles_for_levels(mu, sigma2, levels_needed, **brentq_kwargs)

    coverage = empirical_coverage(y_test, quantile_dict, coverage_levels)
    sharp = sharpness_stats(quantile_dict, sigma2_star)

    scalar = {'rmse': rmse, 'crps': crps, 'nll': nll}
    for a, c in coverage.items():
        scalar[f'coverage_{a}'] = c
    scalar.update(sharp)

    stratified = (stratified_coverage(x_test, y_test, quantile_dict, coverage_levels, n_bins)
                  if include_stratified else [])

    return {'scalar': scalar, 'pit': pit, 'stratified': stratified}


# ============================== Ensemble-size subsampling (Deep Ensemble only) ============

def subsample_members(mu_all: np.ndarray, sigma2_all: np.ndarray, M: int, R: int, seed: int):
    """Yields (replicate_index, member_indices, mu_subset, sigma2_subset) for R random
    subsets of size M drawn without replacement from the K full members (mu_all/
    sigma2_all shape (K, N)). If C(K,M) <= R, enumerates ALL unique subsets instead
    (handles M==K, where only 1 subset exists, without producing R identical rows)."""
    mu_all = np.asarray(mu_all)
    sigma2_all = np.asarray(sigma2_all)
    K = mu_all.shape[0]
    if M > K:
        raise ValueError(f"Requested subset size M={M} exceeds available K={K} members")

    total_subsets = comb(K, M)
    if total_subsets <= R:
        subsets = list(combinations(range(K), M))
    else:
        rng = np.random.default_rng(seed)
        seen = set()
        subsets = []
        while len(subsets) < R:
            idx = tuple(sorted(rng.choice(K, size=M, replace=False).tolist()))
            if idx not in seen:
                seen.add(idx)
                subsets.append(idx)

    for r, idx in enumerate(subsets):
        idx_arr = np.array(idx)
        yield r, idx_arr, mu_all[idx_arr], sigma2_all[idx_arr]
