"""
Exact, analytic EXPECTED scores -- CRPS and NLL (log score) -- of a Gaussian or
Gaussian-mixture forecast against a KNOWN true distribution N(mu_t, sig_t^2).

This is the counterpart to utils/mixture_metrics.py's mixture_crps/mixture_nll, which
score a forecast against one OBSERVED y. Here, since the true DGP is known exactly in
this repo's synthetic experiments, "how good is this forecast" is computed as a real
expectation over the true distribution instead of Monte-Carlo-averaging over a finite
noisy sample -- eliminating sampling variance entirely.

Why this matters (the bug this module fixes): scoring a forecast against a single
deterministic point (e.g. the clean/noise-free mu_true, with zero variance) breaks the
proper-scoring-rule guarantee -- both CRPS and NLL then improve monotonically as the
forecast's own variance shrinks to zero, rewarding overconfidence rather than measuring
genuine predictive accuracy. Scoring against the full true distribution (or, equivalently
here, against its exact expectation) restores properness: the expected score is uniquely
minimized when the forecast equals the truth.

CRPS/NLL asymmetry: a Gaussian-mixture forecast has a CLOSED FORM for its expected CRPS
against a Gaussian truth (Grimit et al. 2006's energy-distance identity, extended here by
folding the truth's own variance into the "against y" term). No closed form exists for a
mixture's expected NLL (the log of a sum of Gaussians doesn't reduce to elementary
functions under expectation) -- so expected_nll_mixture_vs_gaussian uses fixed-node
Gauss-Hermite quadrature instead. This is deterministic (fixed quadrature nodes/weights,
no RNG), not Monte Carlo.

Pure numpy/scipy. No torch, no autograd, no random sampling anywhere in this module.
"""
from typing import Optional

import numpy as np
from numpy.polynomial.hermite import hermgauss

from utils.mixture_metrics import abs_moment_gaussian as _A
from utils.mixture_metrics import mixture_log_pdf

_SIGMA_FLOOR = 1e-12

# Fixed 64-node Gauss-Hermite rule, computed once at import time (deterministic).
_GH_NODES, _GH_WEIGHTS = hermgauss(64)


def _safe_sigma(sig):
    return np.maximum(np.asarray(sig, dtype=float), _SIGMA_FLOOR)


def expected_crps_gaussian_vs_gaussian(mu_f, sig_f, mu_t, sig_t):
    """E_{y~N(mu_t,sig_t^2)}[CRPS(N(mu_f,sig_f^2), y)]
        = A(mu_f - mu_t, sqrt(sig_f^2 + sig_t^2)) - sig_f/sqrt(pi)
    where A(m,s) = E|W| for W ~ N(m, s^2). All args broadcast elementwise; returns an
    array of the broadcast shape."""
    mu_f = np.asarray(mu_f, dtype=float)
    mu_t = np.asarray(mu_t, dtype=float)
    sig_f = _safe_sigma(sig_f)
    sig_t = _safe_sigma(sig_t)
    combined_sigma = np.sqrt(sig_f ** 2 + sig_t ** 2)
    return _A(mu_f - mu_t, combined_sigma) - sig_f / np.sqrt(np.pi)


def expected_crps_mixture_vs_gaussian(weights, mu_k, sig_k, mu_t, sig_t):
    """E_{y~N(mu_t,sig_t^2)}[CRPS(F, y)] for F = sum_k w_k N(mu_k, sig_k^2), via
    Grimit et al. 2006 with the truth's variance folded into the first (single-sum) term
    only:
        E[CRPS] = sum_k w_k * A(mu_k - mu_t, sqrt(sig_k^2 + sig_t^2))
                  - 0.5 * sum_k sum_l w_k w_l * A(mu_k - mu_l, sqrt(sig_k^2 + sig_l^2))

    mu_k/sig_k: shape (M, N) (M mixture components/members, N points). mu_t/sig_t: (N,).
    weights: None -> uniform 1/M (Deep Ensemble members / MCMC draws / dropout passes,
    the equal-weight case used everywhere in this repo); otherwise shape (M,) or (M, N).
    Returns an (N,) array; caller takes .mean() for a scalar summary."""
    mu_k = np.asarray(mu_k, dtype=float)
    sig_k = _safe_sigma(sig_k)
    mu_t = np.asarray(mu_t, dtype=float)
    sig_t = _safe_sigma(sig_t)
    M = mu_k.shape[0]

    if weights is None:
        w = np.full(M, 1.0 / M)
    else:
        w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = w[:, np.newaxis]  # (M, 1), broadcasts over N

    term1 = (w * _A(mu_k - mu_t[np.newaxis, :], np.sqrt(sig_k ** 2 + sig_t[np.newaxis, :] ** 2))).sum(axis=0)

    pair_mu = mu_k[:, np.newaxis, :] - mu_k[np.newaxis, :, :]
    pair_sigma = np.sqrt(sig_k[:, np.newaxis, :] ** 2 + sig_k[np.newaxis, :, :] ** 2)
    pair_w = w[:, np.newaxis, :] * w[np.newaxis, :, :]  # (M,1,*) x (1,M,*) broadcasts to (M,M,N)
    term2 = 0.5 * (pair_w * _A(pair_mu, pair_sigma)).sum(axis=(0, 1))

    return term1 - term2


def expected_nll_gaussian_vs_gaussian(mu_f, sig_f, mu_t, sig_t):
    """Closed-form expected NLL (cross-entropy) of a Gaussian forecast against a Gaussian
    truth:
        E_{y~N(mu_t,sig_t^2)}[-log N(y; mu_f, sig_f^2)]
            = 0.5*log(2*pi*sig_f^2) + (sig_t^2 + (mu_t-mu_f)^2) / (2*sig_f^2)
    """
    mu_f = np.asarray(mu_f, dtype=float)
    mu_t = np.asarray(mu_t, dtype=float)
    sig_f = _safe_sigma(sig_f)
    sig_t = _safe_sigma(sig_t)
    return 0.5 * np.log(2 * np.pi * sig_f ** 2) + (sig_t ** 2 + (mu_t - mu_f) ** 2) / (2 * sig_f ** 2)


def expected_nll_mixture_vs_gaussian(weights, mu_k, sig_k, mu_t, sig_t, n_quad: int = 64):
    """E_{y~N(mu_t,sig_t^2)}[-log f_mix(y)] for a Gaussian-mixture forecast -- NO closed
    form exists (see module docstring), so this uses fixed n_quad-node Gauss-Hermite
    quadrature: substituting y = mu_t + sqrt(2)*sig_t*z reduces the Gaussian-weighted
    integral to sum_i (w_i/sqrt(pi)) * (-log f_mix(mu_t + sqrt(2)*sig_t*z_i)), where
    (z_i, w_i) are the standard Gauss-Hermite nodes/weights. Deterministic, not Monte
    Carlo. mu_k/sig_k: (M, N); mu_t/sig_t: (N,). Returns an (N,) array.

    weights is currently only supported as None (uniform 1/M, matching
    expected_crps_mixture_vs_gaussian's default and mixture_log_pdf's own uniform-weight
    convention) -- non-uniform weighting would require passing weights through to a
    weighted log-sum-exp, not needed by any call site in this repo today."""
    if weights is not None:
        raise NotImplementedError(
            "expected_nll_mixture_vs_gaussian only supports uniform weights (weights=None) "
            "today -- mixture_log_pdf itself assumes equal-weight components."
        )
    mu_k = np.asarray(mu_k, dtype=float)
    sig_k = _safe_sigma(sig_k)
    mu_t = np.asarray(mu_t, dtype=float)
    sig_t = _safe_sigma(sig_t)

    if n_quad == 64:
        nodes, weights_gh = _GH_NODES, _GH_WEIGHTS
    else:
        nodes, weights_gh = hermgauss(n_quad)

    N = mu_t.shape[0]
    total = np.zeros(N)
    for z, w in zip(nodes, weights_gh):
        y_node = mu_t + np.sqrt(2.0) * sig_t * z
        log_f = mixture_log_pdf(y_node, mu_k, sig_k ** 2)
        total += (w / np.sqrt(np.pi)) * (-log_f)
    return total


def oracle_crps(sig_t):
    """Exact CRPS floor achieved by the true forecast N(mu_t, sig_t^2) scored against its
    own distribution: sig_t / sqrt(pi)."""
    sig_t = _safe_sigma(sig_t)
    return sig_t / np.sqrt(np.pi)


def oracle_nll(sig_t):
    """Exact NLL floor achieved by the true forecast: 0.5*log(2*pi*e*sig_t^2). May be
    negative when sig_t < 1/sqrt(2*pi*e) -- that is correct (differential entropy, not a
    probability), not a bug."""
    sig_t = _safe_sigma(sig_t)
    return 0.5 * np.log(2 * np.pi * np.e * sig_t ** 2)


def iqd(expected_crps, sig_t):
    """Integrated quadratic distance: excess CRPS over the oracle floor."""
    return np.asarray(expected_crps, dtype=float) - oracle_crps(sig_t)


def kl_divergence(expected_nll, sig_t):
    """KL(true || forecast): excess NLL over the oracle floor. Gaussian-forecast only --
    for a mixture forecast, pass its (quadrature-based) expected_nll_mixture_vs_gaussian
    output; there's no separate closed form for the divergence itself."""
    return np.asarray(expected_nll, dtype=float) - oracle_nll(sig_t)


def kl_mean(mu_t, sig_f, mu_f):
    """Mean-shift component of KL(N(mu_t,sig_t^2) || N(mu_f,sig_f^2)):
        (mu_t - mu_f)^2 / (2*sig_f^2)
    Carries the effect of a biased forecast mean (e.g. an omitted-variable-bias shift)."""
    mu_t = np.asarray(mu_t, dtype=float)
    mu_f = np.asarray(mu_f, dtype=float)
    sig_f = _safe_sigma(sig_f)
    return (mu_t - mu_f) ** 2 / (2 * sig_f ** 2)


def score_bundle_pointwise(mu_samples, sigma2_samples, mu_star, sigma2_star, mu_true, sigma_true) -> dict:
    """Same computation as score_bundle, but returns one (N,) array per key instead of a
    region-mean scalar -- for callers that want the score AT each x (e.g. a per-grid-point
    CSV export), not just a region/file-level summary. score_bundle is a thin
    mean-of-each-key wrapper around this function; keep the two in sync by construction
    rather than duplicating the formulas.

    mu_samples/sigma2_samples: raw per-member predictions, shape (M,N), already oriented
    via utils.mixture_metrics.normalize_mixture_arrays. mu_star/sigma2_star: the
    moment-matched aggregate, from utils.metrics.compute_predictive_aggregation.
    mu_true/sigma_true: the known true distribution's mean/std at the same N points.

    Returns (N,)-array-valued: crps_mixture, nll_mixture (primary -- full predictive
    distribution), crps_gaussian, nll_gaussian (secondary -- moment-matched single
    Gaussian, clearly labeled), oracle_crps, oracle_nll, iqd, kl, kl_mean, kl_spread."""
    mu_samples = np.asarray(mu_samples, dtype=float)
    sig_samples = np.sqrt(np.maximum(np.asarray(sigma2_samples, dtype=float), _SIGMA_FLOOR))
    mu_star = np.asarray(mu_star, dtype=float)
    sigma_star = np.sqrt(np.maximum(np.asarray(sigma2_star, dtype=float), _SIGMA_FLOOR))
    mu_true = np.asarray(mu_true, dtype=float)
    sigma_true = _safe_sigma(sigma_true)

    crps_mix = expected_crps_mixture_vs_gaussian(None, mu_samples, sig_samples, mu_true, sigma_true)
    nll_mix = expected_nll_mixture_vs_gaussian(None, mu_samples, sig_samples, mu_true, sigma_true)
    crps_gauss = expected_crps_gaussian_vs_gaussian(mu_star, sigma_star, mu_true, sigma_true)
    nll_gauss = expected_nll_gaussian_vs_gaussian(mu_star, sigma_star, mu_true, sigma_true)
    o_crps = oracle_crps(sigma_true)
    o_nll = oracle_nll(sigma_true)
    return {
        'crps_mixture': crps_mix, 'nll_mixture': nll_mix,
        'crps_gaussian': crps_gauss, 'nll_gaussian': nll_gauss,
        'oracle_crps': o_crps, 'oracle_nll': o_nll,
        'iqd': crps_mix - o_crps, 'kl': nll_gauss - o_nll,
        'kl_mean': kl_mean(mu_true, sigma_star, mu_star),
        'kl_spread': kl_spread(sigma_star, sigma_true),
    }


def score_bundle(mu_samples, sigma2_samples, mu_star, sigma2_star, mu_true, sigma_true) -> dict:
    """Convenience wrapper bundling every analytic score this repo's experiment files need
    for one (model, DGP, region/scenario) cell, as REGION-MEAN scalars (mean of
    score_bundle_pointwise's per-point arrays). Single source of truth for the repeating
    "one region's worth of scores" computation used across ood_experiments.py /
    undersampling_experiments.py / sample_size_experiments.py / noise_level_experiments.py /
    ovb_experiments.py -- avoids duplicating this bundle at ~30 call sites. See
    score_bundle_pointwise for the per-point (N,)-array version (used by the raw CSV export
    scripts) and for argument/return-key documentation."""
    pointwise = score_bundle_pointwise(mu_samples, sigma2_samples, mu_star, sigma2_star, mu_true, sigma_true)
    return {k: float(np.mean(v)) for k, v in pointwise.items()}


def kl_spread(sig_f, sig_t):
    """Spread/variance-mismatch component of KL(N(mu_t,sig_t^2) || N(mu_f,sig_f^2)):
        log(sig_f/sig_t) + sig_t^2/(2*sig_f^2) - 0.5
    kl_mean(mu_t, sig_f, mu_f) + kl_spread(sig_f, sig_t) == the full Gaussian-vs-Gaussian
    KL divergence, and equals kl_divergence(expected_nll_gaussian_vs_gaussian(...), sig_t)
    exactly (verified in tests/test_analytic_scores.py). Small kl_spread despite a biased
    mean (large kl_mean) indicates the forecaster widened sigma_f to absorb a structural
    bias rather than correcting mu_f -- the flagship OVB diagnostic."""
    sig_f = _safe_sigma(sig_f)
    sig_t = _safe_sigma(sig_t)
    return np.log(sig_f / sig_t) + sig_t ** 2 / (2 * sig_f ** 2) - 0.5
