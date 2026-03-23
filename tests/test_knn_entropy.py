"""Tests for Kozachenko–Leonenko entropy and k-NN AU/EU decomposition."""

import numpy as np
import pytest

from utils.knn_entropy import (
    entropy_uncertainty_knn_gaussian_mixture,
    kozachenko_leonenko_entropy_1d,
)


def _gaussian_entropy_analytic(variance: float) -> float:
    return 0.5 * np.log(2 * np.pi * np.e * variance)


def test_kl_entropy_matches_gaussian_large_sample():
    rng = np.random.default_rng(42)
    sigma2 = 2.5
    L = 15000
    k = 3
    x = rng.normal(0.0, np.sqrt(sigma2), size=L)
    h_hat = kozachenko_leonenko_entropy_1d(x, k=k)
    h_true = _gaussian_entropy_analytic(sigma2)
    assert np.isfinite(h_hat)
    assert abs(h_hat - h_true) < 0.12


def test_kl_entropy_invalid_returns_nan():
    x = np.array([1.0, 2.0])
    assert np.isnan(kozachenko_leonenko_entropy_1d(x, k=3))


def test_entropy_uncertainty_knn_gaussian_mixture_smoke():
    rng = np.random.default_rng(0)
    M, N = 3, 5
    mu = rng.standard_normal(size=(M, N))
    sigma2 = np.full((M, N), 0.5)
    out = entropy_uncertainty_knn_gaussian_mixture(
        mu, sigma2, L=400, k_nn=3, rng=rng
    )
    assert set(out.keys()) == {"aleatoric", "epistemic", "total"}
    assert out["aleatoric"].shape == (N,)
    assert out["epistemic"].shape == (N,)
    assert out["total"].shape == (N,)
    assert np.all(np.isfinite(out["aleatoric"]))
    assert np.all(np.isfinite(out["epistemic"]))
    assert np.all(np.isfinite(out["total"]))


def test_entropy_uncertainty_knn_raises_when_L_not_greater_than_k():
    mu = np.zeros((2, 3))
    sigma2 = np.ones((2, 3))
    with pytest.raises(ValueError, match="L"):
        entropy_uncertainty_knn_gaussian_mixture(mu, sigma2, L=3, k_nn=3, rng=np.random.default_rng(0))
